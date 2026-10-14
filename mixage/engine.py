"""Moteur de séparation : HT-Demucs (spécialiste voix) via ONNX Runtime.

Pas de PyTorch. Entrée : n'importe quel fichier audio lisible par
libsndfile (mp3, wav, flac, ogg...). Sortie : instrumental = mix - voix.

Le modèle est l'export « core » de htdemucs_ft_vocals : la transformée de
Fourier (STFT / iSTFT) est faite en numpy (voir dsp.py) et non dans le graphe
ONNX, ce qui divise par 8 le nombre d'opérations et rend le modèle utilisable
sur carte graphique (WebGPU, voir backends.py).

Un second modèle, Spleeter (mode rapide, voir spleeter.py), fait le même
travail en quelques secondes sur le processeur avec une qualité moindre.
"""
from __future__ import annotations

import collections
import logging
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

from . import backends, dsp

log = logging.getLogger("mixage")


def log_path() -> Path:
    return user_model_dir().parent / "mixage.log"


def setup_logging() -> Path:
    """Journal dans le dossier utilisateur (utile pour diagnostiquer à distance)."""
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
            h = logging.FileHandler(path, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(h)
            log.setLevel(logging.INFO)
    except Exception:  # noqa: BLE001
        pass
    return path


MODEL_SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * MODEL_SAMPLE_RATE)  # 343 980, figé dans le graphe ONNX
SOURCES = ("drums", "bass", "other", "vocals")
VOCALS_ROW = SOURCES.index("vocals")

ProgressCb = Callable[[float, str], None]  # (fraction 0..1, message)

RELEASES = "https://github.com/davidc4489-crypto/mixage/releases/download/"


class ModelFile(NamedTuple):
    filename: str
    url: str
    size: int


#: Modèle core converti en fp32 complet (tools/convert_fp32.py) à partir de
#: monteslu/htdemucs-ft-webgpu (fp16). En fp16, la variance du prologue
#: déborde sur carte graphique (sortie NaN = silence) ; en fp32 le résultat
#: est exact et le CPU va 30 % plus vite (plus d'émulation fp16).
DEMUCS_MODEL = ModelFile("htdemucs_ft_vocals_fp32.onnx",
                         RELEASES + "model-v1/htdemucs_ft_vocals_fp32.onnx", 174_505_377)
#: Spleeter 2 stems (Deezer, MIT), export ONNX de sherpa-onnx (k2-fsa), renommé.
SPLEETER_MODELS = (
    ModelFile("spleeter_vocals.onnx", RELEASES + "model-v2/spleeter_vocals.onnx", 39_318_336),
    ModelFile("spleeter_accompaniment.onnx", RELEASES + "model-v2/spleeter_accompaniment.onnx", 39_318_343),
)
# compatibilité
MODEL_FILENAME = DEMUCS_MODEL.filename
MODEL_URL = DEMUCS_MODEL.url
MODEL_SIZE_BYTES = DEMUCS_MODEL.size

MODELS = {"demucs": (DEMUCS_MODEL,), "spleeter": SPLEETER_MODELS}
MODEL_LABELS = {"demucs": "HT-Demucs (qualité)", "spleeter": "Spleeter (rapide)"}


class Cancelled(Exception):
    """Levée quand l'utilisateur annule le traitement."""


# --------------------------------------------------------------------------
# Localisation / téléchargement des modèles
# --------------------------------------------------------------------------
def _candidate_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    bundled = getattr(sys, "_MEIPASS", None)          # PyInstaller onefile
    if bundled:
        dirs.append(Path(bundled) / "models")
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent.parent
    dirs.append(exe_dir / "models")
    dirs.append(user_model_dir())
    return dirs


def user_model_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "Mixage" / "models"


def find_model(spec: ModelFile = DEMUCS_MODEL) -> Path | None:
    for d in _candidate_model_dirs():
        p = d / spec.filename
        if p.is_file() and p.stat().st_size == spec.size:
            return p
    return None


def download_model(spec: ModelFile = DEMUCS_MODEL, progress: ProgressCb | None = None,
                   cancel: threading.Event | None = None) -> Path:
    """Télécharge un modèle dans le dossier utilisateur."""
    dest_dir = user_model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "Mixage"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or spec.size)
        done = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if progress:
                progress(done / total,
                         f"Téléchargement du modèle {spec.filename}… {done / 2**20:.0f} / {total / 2**20:.0f} Mo")
    if tmp.stat().st_size != spec.size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Téléchargement du modèle incomplet, réessayez.")
    tmp.replace(dest)
    return dest


def ensure_model(spec: ModelFile = DEMUCS_MODEL, progress: ProgressCb | None = None,
                 cancel: threading.Event | None = None) -> Path:
    found = find_model(spec)
    if found is not None:
        return found
    return download_model(spec, progress, cancel)


def ensure_models(model: str, progress: ProgressCb | None = None,
                  cancel: threading.Event | None = None) -> list[Path]:
    return [ensure_model(spec, progress, cancel) for spec in MODELS[model]]


# --------------------------------------------------------------------------
# Accélération et session ONNX
# --------------------------------------------------------------------------
Device = str  # "auto" | "both" | "gpu" | "cpu"
DEVICES = ("auto", "both", "gpu", "cpu")


def gpu_config_path() -> Path:
    return user_model_dir().parent / "gpu_config.json"


def describe_devices() -> str:
    return backends.describe(backends.load_saved_config(gpu_config_path(), DEMUCS_MODEL.filename))


def default_threads() -> int:
    return backends.default_threads()


def _base_options(threads: int | None,
                  level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                  ) -> ort.SessionOptions:
    return backends.base_options(threads, level)


_LEVELS = backends.LEVELS


class Separator:
    """Sessions ONNX (carte graphique et/ou processeur) pour HT-Demucs.

    ``device`` :
      - "auto" : carte graphique si elle donne un résultat juste ; processeur en
        renfort si la machine a assez de mémoire (``HYBRID_MIN_RAM_GB``) ;
      - "both" : carte graphique et processeur ensemble ;
      - "gpu"  : carte graphique seule (erreur s'il n'y en a pas) ;
      - "cpu"  : processeur seul.

    Sur carte graphique, le premier morceau est recalculé sur le processeur et
    comparé : si l'écart dépasse ``VERIFY_TOLERANCE``, la configuration suivante
    (voir backends.gpu_configs) est essayée ; si aucune ne convient, tout le
    reste se fait sur le processeur. La configuration validée est mémorisée.
    """

    #: Délai maximal pour le premier morceau sur carte graphique (compilation
    #: des noyaux incluse), puis pour les suivants. Au-delà : repli CPU.
    GPU_FIRST_TIMEOUT_S = 150.0
    GPU_TIMEOUT_S = 30.0
    #: Écart relatif toléré entre carte graphique et CPU sur le premier morceau.
    VERIFY_TOLERANCE = 0.05
    #: En dessous de ce niveau RMS, un morceau est jugé trop silencieux pour servir
    #: de référence de vérification.
    VERIFY_MIN_RMS = 0.01
    #: Mémoire vive minimale pour faire travailler processeur et carte ensemble
    #: (deux sessions : pic transitoire ~3 Go chacune).
    HYBRID_MIN_RAM_GB = 12.0

    def __init__(self, model_path: Path, device: Device = "auto",
                 threads: int | None = None, verify: bool = True) -> None:
        if device not in DEVICES:
            raise ValueError(f"device inconnu : {device!r}")
        self.model_path = Path(model_path)
        self.threads = threads
        self.verify = verify
        self.mode = device
        self.notes: list[str] = []
        self.session: ort.InferenceSession | None = None      # session GPU courante
        self._cpu_session: ort.InferenceSession | None = None
        self.on_gpu = False
        self.gpu_config: dict | None = None
        self.gpu_runs = 0
        self.gpu_time = 0.0          # durée du dernier morceau GPU
        self.cpu_time = 0.0          # durée d'un morceau CPU (référence)
        self._pending: list[dict] = []
        self._gpu_name = ""
        self._lock = threading.Lock()

        if device != "cpu":
            self._start_gpu()
        if not self.on_gpu:
            self._ensure_cpu_session()

    # -- description --------------------------------------------------------
    @property
    def device_used(self) -> str:
        cpu = f"Processeur ({self.threads or default_threads()} threads)"
        if not self.on_gpu:
            return cpu
        gpu = f"Carte graphique ({backends.backend_label(self.gpu_config)}"
        gpu += f" – {self._gpu_name})" if self._gpu_name else ")"
        return gpu + " + processeur" if self.hybrid else gpu

    @property
    def hybrid(self) -> bool:
        """Processeur et carte graphique travaillent ensemble."""
        if not self.on_gpu or self.mode == "gpu":
            return False
        if self.mode == "both":
            return True
        return backends.total_ram_gb() >= self.HYBRID_MIN_RAM_GB

    # -- création -----------------------------------------------------------
    def _start_gpu(self) -> None:
        self._pending = backends.gpu_configs()
        if not self._pending:
            _, err = backends.webgpu_status()
            return self._no_gpu(f"aucun accélérateur graphique utilisable ({err})")
        gpus = backends.list_gpus()
        if sys.platform == "win32" and not gpus:
            return self._no_gpu("aucune carte graphique physique détectée")
        self._gpu_name = gpus[0] if gpus else ""
        saved = backends.load_saved_config(gpu_config_path(), DEMUCS_MODEL.filename)
        if saved is not None:
            self._pending = [c for c in self._pending if c != saved]
            self._pending.insert(0, saved)
        if not self._open_next_gpu_config():
            self._no_gpu("carte graphique inutilisable (aucune configuration n'a pu démarrer)")

    def _no_gpu(self, msg: str) -> None:
        if self.mode == "gpu":
            raise RuntimeError("Carte graphique demandée mais " + msg + ".")
        self.notes.append(msg)
        log.info("GPU indisponible : %s", msg)

    def _open_next_gpu_config(self) -> bool:
        """Ouvre la prochaine configuration GPU candidate. False si plus aucune."""
        while self._pending:
            cfg = self._pending.pop(0)
            try:
                self.session = backends.open_gpu_session(self.model_path, cfg, self.threads)
                self.on_gpu = True
                self.gpu_config = cfg
                log.info("session GPU ouverte : %s, niveau %s, options %r",
                         backends.backend_label(cfg), cfg["level"], cfg["options"])
                return True
            except Exception as exc:  # noqa: BLE001
                log.info("configuration GPU refusée (%s, %s, %r) : %s",
                         cfg["backend"], cfg["level"], cfg["options"], exc)
        return False

    def _ensure_cpu_session(self) -> None:
        if self._cpu_session is None:
            self._cpu_session = backends.open_cpu_session(self.model_path, self.threads)
            log.info("session CPU créée (%d threads)", self.threads or default_threads())

    def _fallback(self, reason: str) -> None:
        with self._lock:
            if not self.on_gpu:
                return
            log.warning("repli CPU : %s", reason)
            self.notes.append(reason + ", repli sur le processeur")
            backends.save_config(gpu_config_path(), DEMUCS_MODEL.filename, None)
            self.session = None
            self.on_gpu = False
        self._ensure_cpu_session()

    # -- inférence ----------------------------------------------------------
    @staticmethod
    def _infer_raw(session: ort.InferenceSession, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, xt = session.run(["x", "xt"], dsp.model_inputs(chunk))
        return x[0, VOCALS_ROW], xt[0, VOCALS_ROW]

    def run_cpu(self, x: np.ndarray) -> np.ndarray:
        """Voix (2, N) calculée sur le processeur."""
        self._ensure_cpu_session()
        t0 = time.time()
        gx, gxt = self._infer_raw(self._cpu_session, x)
        self.cpu_time = time.time() - t0
        return dsp.stem_from_outputs(gx, gxt, x.shape[-1])

    def run_gpu(self, x: np.ndarray) -> np.ndarray:
        """Voix (2, N) calculée sur la carte graphique (lève si résultat invalide)."""
        session = self.session
        if session is None:
            raise RuntimeError("pas de session GPU")
        t0 = time.time()
        gx, gxt = self._infer_raw(session, x)
        if not (np.isfinite(gx).all() and np.isfinite(gxt).all()):
            raise ArithmeticError("résultat invalide sur la carte graphique (NaN)")
        self.gpu_time = time.time() - t0
        self.gpu_runs += 1
        return dsp.stem_from_outputs(gx, gxt, x.shape[-1])

    def _run_gpu_guarded(self, x: np.ndarray, cancel: threading.Event | None,
                         timeout: float) -> tuple[tuple[np.ndarray, np.ndarray] | None, str | None]:
        """Exécute sur le GPU dans un thread surveillé. Retourne (résultat brut, erreur)."""
        result: dict = {}
        session = self.session

        def worker() -> None:
            try:
                result["y"] = self._infer_raw(session, x)
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        th = threading.Thread(target=worker, daemon=True)
        t0 = time.time()
        th.start()
        while th.is_alive():
            th.join(0.25)
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if time.time() - t0 > timeout:
                return None, f"carte graphique trop lente (plus de {timeout:.0f} s pour un morceau)"
        if "err" in result:
            return None, f"échec sur la carte graphique ({type(result['err']).__name__}: {result['err']})"
        return result["y"], None

    def _verify_first_chunk(self, x: np.ndarray, cancel: threading.Event | None) -> np.ndarray:
        """Premier morceau : compare chaque configuration GPU au CPU, garde la bonne."""
        self._ensure_cpu_session()
        t0 = time.time()
        ref_x, ref_xt = self._infer_raw(self._cpu_session, x)
        self.cpu_time = time.time() - t0
        ref = dsp.stem_from_outputs(ref_x, ref_xt, x.shape[-1])
        log.info("référence CPU du premier morceau : %.1f s", self.cpu_time)
        tried = 0
        while self.on_gpu:
            tried += 1
            t0 = time.time()
            raw, err = self._run_gpu_guarded(x, cancel, self.GPU_FIRST_TIMEOUT_S)
            dt = time.time() - t0
            if raw is not None:
                gx, gxt = raw
                y = dsp.stem_from_outputs(gx, gxt, x.shape[-1])
                finite = bool(np.isfinite(gx).all() and np.isfinite(gxt).all())
                # dénominateur plancher : 1 % de l'énergie du mix, pour ne pas
                # rejeter un GPU juste sur un passage sans voix
                denom = max(float(np.linalg.norm(ref)), 0.01 * float(np.linalg.norm(x)), 1e-9)
                rel = float(np.linalg.norm(y - ref) / denom) if finite else float("inf")
                rel_x = float(np.linalg.norm(gx - ref_x) / (np.linalg.norm(ref_x) + 1e-9)) if finite else float("inf")
                rel_xt = float(np.linalg.norm(gxt - ref_xt) / (np.linalg.norm(ref_xt) + 1e-9)) if finite else float("inf")
                log.info("vérification GPU (%s, %s, %r) : %.1f s, fini=%s, écart voix %.3g "
                         "(spectre %.3g, temporel %.3g ; rms gpu %.3g / cpu %.3g)",
                         self.gpu_config["backend"], self.gpu_config["level"], self.gpu_config["options"],
                         dt, finite, rel, rel_x, rel_xt,
                         float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(ref ** 2))))
                if rel < self.VERIFY_TOLERANCE:
                    self.gpu_runs += 1
                    self.gpu_time = dt
                    backends.save_config(gpu_config_path(), DEMUCS_MODEL.filename, self.gpu_config)
                    if tried > 1:
                        self.notes.append(f"carte graphique validée après {tried} configurations "
                                          f"({backends.backend_label(self.gpu_config)}, niveau "
                                          f"{self.gpu_config['level']})")
                    return y
                err = f"résultat faux sur la carte graphique (écart {rel:.0%} vs processeur)"
            log.warning("configuration GPU écartée : %s", err)
            self.session = None
            if not self._open_next_gpu_config():
                self.on_gpu = False
                self._fallback_after_verify(f"{err} ; {tried} configuration(s) essayée(s)")
                return ref
        return ref

    def _fallback_after_verify(self, reason: str) -> None:
        log.warning("repli CPU : %s", reason)
        self.notes.append(reason + ", repli sur le processeur")
        backends.save_config(gpu_config_path(), DEMUCS_MODEL.filename, None)
        self.session = None
        self.on_gpu = False

    def run(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Voix estimée (2, N) pour un segment (2, N), séquentiellement.

        Les premiers morceaux passent ici (vérification de la carte) ; le reste
        d'une chanson passe par ``separate_vocals`` qui répartit le travail.
        """
        if not self.on_gpu:
            return self.run_cpu(x)
        if self.gpu_runs == 0 and self.verify:
            if float(np.sqrt(np.mean(x ** 2))) < self.VERIFY_MIN_RMS:
                # Morceau quasi silencieux (intro) : comparaison peu fiable,
                # on le calcule sur le CPU et on vérifiera le prochain.
                log.info("morceau quasi silencieux : calcul CPU, vérification GPU reportée")
                return self.run_cpu(x)
            return self._verify_first_chunk(x, cancel)
        timeout = self.GPU_FIRST_TIMEOUT_S if self.gpu_runs == 0 else self.GPU_TIMEOUT_S
        t0 = time.time()
        raw, err = self._run_gpu_guarded(x, cancel, timeout)
        if raw is None:
            self._fallback(err or "erreur inconnue")
            return self.run_cpu(x)
        gx, gxt = raw
        if not (np.isfinite(gx).all() and np.isfinite(gxt).all()):
            self._fallback("résultat invalide sur la carte graphique (NaN)")
            return self.run_cpu(x)
        self.gpu_runs += 1
        self.gpu_time = time.time() - t0
        log.info("morceau GPU %d : %.1f s", self.gpu_runs, self.gpu_time)
        return dsp.stem_from_outputs(gx, gxt, x.shape[-1])

    def close(self) -> None:
        self.session = None
        self._cpu_session = None


def make_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    """Compatibilité : session CPU seule."""
    sep = Separator(model_path, "cpu", threads)
    return sep._cpu_session  # noqa: SLF001


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Retourne (audio stéréo float32 (2, n) au taux natif, taux natif)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.tile(audio, (2, 1))
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    out = soxr.resample(audio.T, src_sr, dst_sr, quality="HQ").T
    return np.ascontiguousarray(out, dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray, sr: int, bitrate_kbps: int = 192) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    if path.suffix.lower() == ".mp3":
        import lameenc
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate_kbps)
        enc.set_in_sample_rate(sr)
        enc.set_channels(audio.shape[0])
        enc.set_quality(2)
        pcm = (audio.T * 32767.0).astype(np.int16).tobytes()
        path.write_bytes(enc.encode(pcm) + enc.flush())
    else:
        sf.write(str(path), audio.T, sr, subtype="PCM_16")


# --------------------------------------------------------------------------
# Séparation HT-Demucs
# --------------------------------------------------------------------------
def _transition_window(segment: int, overlap_frac: float = 0.25) -> np.ndarray:
    t = int(segment * overlap_frac)
    w = np.ones(segment, dtype=np.float32)
    fade = np.linspace(0, 1, t, dtype=np.float32)
    w[:t] = fade
    w[-t:] = fade[::-1]
    return w


class _Scheduler:
    """Répartit les morceaux restants entre la carte graphique et le processeur.

    La carte prend les morceaux dans l'ordre ; le processeur (si ``hybrid``)
    prend par la fin, et seulement si la carte n'aurait pas fini avant lui
    (``restants × durée GPU > durée CPU``). Un morceau trop long sur la carte
    est redonné au processeur et la carte est abandonnée.
    """

    def __init__(self, sep: Separator, chunks: list[np.ndarray], indices: list[int],
                 cancel: threading.Event | None) -> None:
        self.sep = sep
        self.chunks = chunks
        self.pending: collections.deque[int] = collections.deque(indices)
        self.done: dict[int, np.ndarray] = {}
        self.cancel = cancel
        self.lock = threading.Lock()
        self.gpu_alive = sep.on_gpu
        self.gpu_idx: int | None = None
        self.gpu_since = 0.0
        self.cpu_active = False

    # -- travailleurs ------------------------------------------------------
    def _gpu_worker(self) -> None:
        while True:
            with self.lock:
                if not self.gpu_alive or not self.pending or self._cancelled():
                    return
                idx = self.pending.popleft()
                self.gpu_idx, self.gpu_since = idx, time.time()
            err: Exception | None = None
            y = None
            try:
                y = self.sep.run_gpu(self.chunks[idx])
            except Exception as exc:  # noqa: BLE001
                err = exc
            with self.lock:
                if self.gpu_idx != idx:
                    continue                      # abandonné entre-temps (délai dépassé)
                self.gpu_idx = None
                if err is not None:
                    self.pending.appendleft(idx)
                    self.gpu_alive = False
                    reason = f"échec sur la carte graphique ({type(err).__name__}: {err})"
                else:
                    self.done[idx] = y
                    log.info("morceau GPU %d (n° %d) : %.1f s", self.sep.gpu_runs, idx + 1, self.sep.gpu_time)
                    continue
            self.sep._fallback(reason)  # noqa: SLF001

    def _cpu_worker(self) -> None:
        while True:
            with self.lock:
                if self._cancelled():
                    return
                if not self.pending:
                    if self.gpu_idx is None:
                        return                    # tout est distribué et terminé
                    take = False                  # la carte finit le dernier ; attendre au cas où
                else:
                    gpu_busy = self.gpu_alive
                    take = (not gpu_busy or not self.sep.gpu_time or not self.sep.cpu_time
                            or len(self.pending) * self.sep.gpu_time > self.sep.cpu_time)
                    if take:
                        idx = self.pending.pop()
            if not take:
                time.sleep(0.2)
                continue
            y = self.sep.run_cpu(self.chunks[idx])
            with self.lock:
                self.done[idx] = y
            log.info("morceau CPU n° %d : %.1f s", idx + 1, self.sep.cpu_time)

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    # -- pilotage ----------------------------------------------------------
    def run(self, progress: Callable[[int], None]) -> dict[int, np.ndarray]:
        n_total = len(self.pending)
        threads: list[threading.Thread] = []
        if self.gpu_alive:
            threads.append(threading.Thread(target=self._gpu_worker, daemon=True, name="gpu"))
        hybrid = self.sep.hybrid or not self.gpu_alive
        if hybrid:
            self.cpu_active = True
            threads.append(threading.Thread(target=self._cpu_worker, daemon=True, name="cpu"))
        for th in threads:
            th.start()
        last = -1
        while True:
            time.sleep(0.2)
            if self._cancelled():
                raise Cancelled()
            with self.lock:
                n_done = len(self.done)
                finished = n_done >= n_total
                # carte graphique bloquée ?
                if self.gpu_idx is not None and time.time() - self.gpu_since > self.sep.GPU_TIMEOUT_S:
                    idx, self.gpu_idx = self.gpu_idx, None
                    self.pending.appendleft(idx)
                    self.gpu_alive = False
                    timeout_reason = (f"carte graphique trop lente (plus de "
                                      f"{self.sep.GPU_TIMEOUT_S:.0f} s pour un morceau)")
                else:
                    timeout_reason = None
                need_cpu = (not self.gpu_alive) and not self.cpu_active and not finished
                if need_cpu:
                    self.cpu_active = True
            if timeout_reason:
                self.sep._fallback(timeout_reason)  # noqa: SLF001
            if need_cpu:
                th = threading.Thread(target=self._cpu_worker, daemon=True, name="cpu")
                th.start()
                threads.append(th)
            if n_done != last:
                progress(n_done)
                last = n_done
            if finished:
                break
        return self.done


def separate_vocals(sep: Separator, mix: np.ndarray, *,
                    progress: ProgressCb | None = None,
                    cancel: threading.Event | None = None) -> np.ndarray:
    """Voix estimée, (2, n) à 44,1 kHz. Découpage 7,8 s avec recouvrement 25 %."""
    total = mix.shape[1]
    overlap = N_SAMPLES // 4
    stride = N_SAMPLES - overlap
    n_chunks = max(1, (total + stride - 1) // stride)
    window = _transition_window(N_SAMPLES)

    chunks: list[np.ndarray] = []
    for i in range(n_chunks):
        start = i * stride
        chunk = mix[:, start:start + N_SAMPLES]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])))
        chunks.append(np.ascontiguousarray(chunk, dtype=np.float32))

    results: dict[int, np.ndarray] = {}

    def report(n_done: int) -> None:
        if progress:
            where = sep.device_used
            extra = ""
            if n_done == 0 and sep.on_gpu:
                extra = " (première passe : vérification de la carte graphique)"
            progress(n_done / n_chunks,
                     f"Séparation… morceau {min(n_done + 1, n_chunks)}/{n_chunks} sur {where}{extra}")

    # Premiers morceaux séquentiels, le temps de valider la carte graphique.
    i = 0
    while i < n_chunks and (not sep.on_gpu or sep.gpu_runs == 0):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        report(i)
        t0 = time.time()
        results[i] = sep.run(chunks[i], cancel)
        if i == 0:
            log.info("premier morceau : %.1f s sur %s", time.time() - t0, sep.device_used)
        i += 1
        if not sep.on_gpu and i < n_chunks:
            break                                   # le reste part dans l'ordonnanceur (CPU)

    if i < n_chunks:
        sched = _Scheduler(sep, chunks, list(range(i, n_chunks)), cancel)
        log.info("répartition de %d morceaux : %s", n_chunks - i, sep.device_used)
        results.update(sched.run(lambda d: report(i + d)))

    out = np.zeros((2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    for k in range(n_chunks):
        start = k * stride
        end = min(start + N_SAMPLES, total)
        n = end - start
        out[:, start:end] += results[k][:, :n] * window[:n]
        weight[start:end] += window[:n]
    return out / np.maximum(weight, 1e-8)


# --------------------------------------------------------------------------
# Pipeline complet
# --------------------------------------------------------------------------
def output_path_for(src: Path, fmt: str = "wav") -> Path:
    return src.with_name(f"{src.stem} (instrumental).{fmt}")


def make_instrumental(src: Path, dst: Path | None = None, *, fmt: str = "wav",
                      model: str = "demucs",
                      device: Device = "auto",
                      threads: int | None = None,
                      progress: ProgressCb | None = None,
                      cancel: threading.Event | None = None,
                      info: dict | None = None) -> Path:
    """Pipeline complet : fichier audio -> fichier instrumental (sans voix).

    ``model`` : "demucs" (qualité) ou "spleeter" (rapide, processeur).
    ``info`` (optionnel) reçoit "device" (périphérique utilisé), "notes"
    (diagnostics) et "seconds" (durée du traitement).
    """
    if model not in MODELS:
        raise ValueError(f"modèle inconnu : {model!r}")
    t_start = time.time()
    setup_logging()
    log.info("début : %s -> %s (modèle=%s, device=%s, fmt=%s)", src, dst, model, device, fmt)
    src = Path(src)
    dst = Path(dst) if dst else output_path_for(src, fmt)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.0, "Préparation du modèle…")
    paths = ensure_models(model, lambda f, m: report(0.05 * f, m), cancel)
    report(0.05, "Chargement du modèle (quelques secondes)…")
    notes: list[str] = []
    if model == "spleeter":
        from .spleeter import Spleeter
        sep_fast = Spleeter(paths, threads)
        device_used = sep_fast.device_used
    else:
        sep = Separator(paths[0], device, threads)
        device_used = sep.device_used
    report(0.08, f"Calcul sur : {device_used}")

    report(0.10, "Lecture du fichier audio…")
    native, sr = load_audio(src)
    mix = resample(native, sr, MODEL_SAMPLE_RATE)

    if model == "spleeter":
        vocals = sep_fast.vocals(mix, progress=lambda f, m: report(0.10 + 0.85 * f, m), cancel=cancel)
    else:
        vocals = separate_vocals(sep, mix, progress=lambda f, m: report(0.10 + 0.85 * f, m), cancel=cancel)
        notes = list(sep.notes)
        device_used = sep.device_used
        sep.close()
    if info is not None:
        info["device"] = device_used
        info["notes"] = notes
        info["model"] = model

    report(0.95, "Écriture du fichier…")
    vocals_native = resample(vocals, MODEL_SAMPLE_RATE, sr)
    n = min(native.shape[1], vocals_native.shape[1])
    instrumental = native[:, :n] - vocals_native[:, :n]
    write_audio(dst, instrumental, sr)
    if info is not None:
        info["seconds"] = time.time() - t_start
    log.info("terminé en %.0f s sur %s (%s)", time.time() - t_start, device_used, model)
    report(1.0, "Terminé")
    return dst


def speed_test(progress: ProgressCb | None = None,
               cancel: threading.Event | None = None) -> str:
    """Chronomètre un morceau de 7,8 s : carte graphique, processeur, et Spleeter."""
    setup_logging()
    lines = [describe_devices(), ""]
    model = ensure_model(DEMUCS_MODEL, progress, cancel)
    # signal de test déterministe : deux sinus + bruit (évite un vrai fichier)
    t = np.arange(N_SAMPLES, dtype=np.float32) / MODEL_SAMPLE_RATE
    rng = np.random.default_rng(0)
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)])
    x = (x + 0.02 * rng.standard_normal(x.shape)).astype(np.float32)
    results: dict[str, np.ndarray] = {}
    for device, label in (("gpu", "Carte graphique"), ("cpu", "Processeur")):
        if progress:
            progress(0.5 if device == "cpu" else 0.1, f"Test {label}…")
        try:
            sep = Separator(model, device, verify=False)
            t0 = time.time()
            y = sep.run(x, cancel)              # première passe (compilation des noyaux)
            first = time.time() - t0
            t0 = time.time()
            y = sep.run(x, cancel)
            second = time.time() - t0
            results[device] = y
            if sep.notes:
                lines.append(f"{label} : {'; '.join(sep.notes)}")
            else:
                lines.append(f"{sep.device_used} : 1er morceau {first:.1f} s, "
                             f"2e morceau {second:.1f} s")
            sep.close()
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{label} : indisponible ({type(exc).__name__}: {exc})")
    if "gpu" in results and "cpu" in results:
        g, c = results["gpu"], results["cpu"]
        if not np.isfinite(g).all():
            lines.append("Comparaison : la carte graphique renvoie des NaN (résultat invalide).")
        else:
            rel = float(np.linalg.norm(g - c) / (np.linalg.norm(c) + 1e-9))
            verdict = "identique" if rel < 1e-2 else ("proche" if rel < 1e-1 else "DIFFÉRENT")
            lines.append(f"Comparaison carte/processeur : écart relatif {rel:.1e} ({verdict}).")
    try:
        if progress:
            progress(0.8, "Test Spleeter (mode rapide)…")
        from .spleeter import Spleeter
        fast = Spleeter(ensure_models("spleeter", progress, cancel))
        t0 = time.time()
        fast.vocals(x)
        lines.append(f"Spleeter (mode rapide, processeur) : {time.time() - t0:.1f} s pour 7,8 s d'audio")
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Spleeter : indisponible ({type(exc).__name__}: {exc})")
    lines += ["", f"Journal : {log_path()}"]
    text = "\n".join(lines)
    log.info("test de vitesse :\n%s", text)
    return text
