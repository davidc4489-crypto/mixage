"""Moteur de séparation : HT-Demucs (spécialiste voix) via ONNX Runtime.

Pas de PyTorch. Entrée : n'importe quel fichier audio lisible par
libsndfile (mp3, wav, flac, ogg...). Sortie : instrumental = mix - voix.

Le modèle est l'export « core » de htdemucs_ft_vocals : la transformée de
Fourier (STFT / iSTFT) est faite en numpy (voir dsp.py) et non dans le graphe
ONNX, ce qui divise par 8 le nombre d'opérations et rend le modèle utilisable
sur carte graphique (DirectML).

Réglages mémoire importants (mesurés sur un PC 8 cœurs / 15 Go) :
- arena mémoire + mem_pattern activés (défaut ORT)  -> 5 à 7 Go, plantage possible
- sans arena ni mem_pattern                          -> ~0,4 Go
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import logging

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

from . import dsp

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
    except Exception:
        pass
    return path

MODEL_SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * MODEL_SAMPLE_RATE)  # 343 980, figé dans le graphe ONNX
SOURCES = ("drums", "bass", "other", "vocals")
VOCALS_ROW = SOURCES.index("vocals")

#: Modèle core converti en fp32 complet (tools/convert_fp32.py) à partir de
#: monteslu/htdemucs-ft-webgpu (fp16). En fp16, la variance du prologue
#: déborde sur carte graphique (sortie NaN = silence) ; en fp32 le résultat
#: est exact et le CPU va 30 % plus vite (plus d'émulation fp16).
MODEL_FILENAME = "htdemucs_ft_vocals_fp32.onnx"
MODEL_URL = ("https://github.com/davidc4489-crypto/mixage/releases/download/"
             "model-v1/" + MODEL_FILENAME)
MODEL_SIZE_BYTES = 174_505_377

ProgressCb = Callable[[float, str], None]  # (fraction 0..1, message)


class Cancelled(Exception):
    """Levée quand l'utilisateur annule le traitement."""


# --------------------------------------------------------------------------
# Localisation / téléchargement du modèle
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


def find_model() -> Path | None:
    for d in _candidate_model_dirs():
        p = d / MODEL_FILENAME
        if p.is_file() and p.stat().st_size == MODEL_SIZE_BYTES:
            return p
    return None


def download_model(progress: ProgressCb | None = None,
                   cancel: threading.Event | None = None) -> Path:
    """Télécharge le modèle (165 Mo) dans le dossier utilisateur."""
    dest_dir = user_model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / MODEL_FILENAME
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mixage"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or MODEL_SIZE_BYTES)
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
                         f"Téléchargement du modèle… {done / 2**20:.0f} / {total / 2**20:.0f} Mo")
    if tmp.stat().st_size != MODEL_SIZE_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Téléchargement du modèle incomplet, réessayez.")
    tmp.replace(dest)
    return dest


def ensure_model(progress: ProgressCb | None = None,
                 cancel: threading.Event | None = None) -> Path:
    found = find_model()
    if found is not None:
        return found
    return download_model(progress, cancel)


# --------------------------------------------------------------------------
# Accélération matérielle (carte graphique) et session ONNX
# --------------------------------------------------------------------------
#: Fournisseurs GPU connus d'ONNX Runtime, par ordre de préférence.
#: DirectML (Windows) couvre NVIDIA, AMD et Intel via DirectX 12.
GPU_PROVIDERS = (
    ("DmlExecutionProvider", "DirectML"),
    ("CUDAExecutionProvider", "CUDA"),
    ("ROCMExecutionProvider", "ROCm"),
    ("CoreMLExecutionProvider", "CoreML"),
)
#: Adaptateurs logiciels Windows à ignorer : pas de vraie carte graphique.
_SOFTWARE_ADAPTERS = ("basic display", "basic render", "remote display", "warp", "virtual")

Device = str  # "auto" | "gpu" | "cpu"


def list_gpus() -> list[str]:
    """Noms des cartes graphiques physiques (Windows uniquement, sinon [])."""
    if sys.platform != "win32":
        return []
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception:
        return []
    names = [n.strip() for n in out.splitlines() if n.strip()]
    return [n for n in names if not any(k in n.lower() for k in _SOFTWARE_ADAPTERS)]


def available_gpu_providers() -> list[tuple[str, str]]:
    avail = set(ort.get_available_providers())
    return [(p, label) for p, label in GPU_PROVIDERS if p in avail]


def describe_devices() -> str:
    """Texte de diagnostic : fournisseurs disponibles et cartes détectées."""
    provs = ", ".join(label for _, label in available_gpu_providers()) or "aucun"
    gpus = ", ".join(list_gpus()) or ("non détectée" if sys.platform == "win32" else "n/a hors Windows")
    saved = _load_saved_gpu_config()
    cfg = (f"niveau {saved['level']}, options {saved['options']}" if saved
           else "aucune encore validée")
    return (f"ONNX Runtime {ort.__version__}\n"
            f"Accélérateurs disponibles : {provs}\n"
            f"Cartes graphiques : {gpus}\n"
            f"Configuration carte graphique mémorisée : {cfg}")


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n, 4))


def _base_options(threads: int | None,
                  level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                  ) -> ort.SessionOptions:
    so = ort.SessionOptions()
    # Mesuré sur le modèle core : l'arena mémoire + mem_pattern font monter la
    # RAM à 5-6 Go ; sans eux, 0,4 Go avec les optimisations complètes
    # (mem_pattern doit de toute façon être désactivé avec DirectML).
    so.graph_optimization_level = level
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads or default_threads()
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


_LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


def _gpu_configs(provider: str) -> list[dict]:
    """Configurations à essayer, dans l'ordre, pour un fournisseur GPU.

    Chaque configuration est vérifiée sur le premier morceau contre le CPU ;
    la première qui donne un résultat juste est conservée (et mémorisée).
    Les « metacommands » DirectML sont des noyaux fournis par le pilote de la
    carte : rapides, mais source connue de résultats faux sur certains pilotes.
    """
    if provider == "DmlExecutionProvider":
        return [
            {"provider": provider, "level": "all", "options": {"device_id": 0}},
            {"provider": provider, "level": "all", "options": {"device_id": 0, "disable_metacommands": "1"}},
            {"provider": provider, "level": "basic", "options": {"device_id": 0, "disable_metacommands": "1"}},
            {"provider": provider, "level": "disable", "options": {"device_id": 0, "disable_metacommands": "1"}},
            {"provider": provider, "level": "disable", "options": {}},
        ]
    return [{"provider": provider, "level": "all", "options": {}},
            {"provider": provider, "level": "disable", "options": {}}]


def gpu_config_path() -> Path:
    return user_model_dir().parent / "gpu_config.json"


def _load_saved_gpu_config() -> dict | None:
    try:
        import json
        cfg = json.loads(gpu_config_path().read_text(encoding="utf-8"))
        if cfg.get("model") == MODEL_FILENAME and cfg.get("ort") == ort.__version__:
            return cfg["config"]
    except Exception:
        pass
    return None


def _save_gpu_config(cfg: dict | None) -> None:
    try:
        import json
        gpu_config_path().parent.mkdir(parents=True, exist_ok=True)
        if cfg is None:
            gpu_config_path().unlink(missing_ok=True)
        else:
            gpu_config_path().write_text(json.dumps(
                {"model": MODEL_FILENAME, "ort": ort.__version__, "config": cfg},
                ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


class Separator:
    """Session ONNX avec choix du périphérique et repli automatique sur le CPU.

    ``device`` : "auto" (carte graphique si possible), "gpu" (carte graphique
    obligatoire, erreur sinon) ou "cpu".

    Sur carte graphique, le premier morceau est recalculé sur le CPU et comparé :
    si l'écart dépasse ``VERIFY_TOLERANCE``, la configuration suivante est
    essayée ; si aucune ne convient, tout le reste se fait sur le CPU.
    """

    #: Délai maximal pour le premier morceau sur carte graphique (compilation
    #: des opérateurs incluse), puis pour les suivants. Au-delà : repli CPU.
    GPU_FIRST_TIMEOUT_S = 150.0
    GPU_TIMEOUT_S = 25.0
    #: Écart relatif toléré entre carte graphique et CPU sur le premier morceau.
    VERIFY_TOLERANCE = 0.05
    #: En dessous de ce niveau RMS, un morceau est jugé trop silencieux pour servir
    #: de référence de vérification.
    VERIFY_MIN_RMS = 0.01

    def __init__(self, model_path: Path, device: Device = "auto",
                 threads: int | None = None, verify: bool = True) -> None:
        self.model_path = Path(model_path)
        self.threads = threads
        self.verify = verify
        self.session: ort.InferenceSession | None = None
        self._cpu_session: ort.InferenceSession | None = None
        self.device_used = ""
        self.on_gpu = False
        self.notes: list[str] = []
        self.gpu_runs = 0
        self.gpu_config: dict | None = None
        self._pending: list[dict] = []
        self._gpu_label = ""
        self._gpu_name = ""

        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device inconnu : {device!r}")
        self._required = device == "gpu"
        if device != "cpu":
            self._start_gpu()
        if self.session is None:
            self._use_cpu()

    # -- création ----------------------------------------------------------
    def _start_gpu(self) -> None:
        providers = available_gpu_providers()
        if not providers:
            return self._no_gpu("aucun accélérateur graphique disponible dans cette version")
        gpus = list_gpus()
        if sys.platform == "win32" and not gpus:
            return self._no_gpu("aucune carte graphique physique détectée")
        self._gpu_name = gpus[0] if gpus else ""
        self._pending = []
        for provider, label in providers:
            self._pending += [dict(c, label=label) for c in _gpu_configs(provider)]
        saved = _load_saved_gpu_config()
        if saved is not None:
            self._pending = [c for c in self._pending if c != saved]
            self._pending.insert(0, saved)
        if not self._open_next_gpu_config():
            self._no_gpu("carte graphique inutilisable (aucune configuration n'a pu démarrer)")

    def _no_gpu(self, msg: str) -> None:
        if self._required:
            raise RuntimeError("Carte graphique demandée mais " + msg + ".")
        self.notes.append(msg)
        log.info("GPU indisponible : %s", msg)

    def _open_next_gpu_config(self) -> bool:
        """Ouvre la prochaine configuration GPU candidate. False si plus aucune."""
        while self._pending:
            cfg = self._pending.pop(0)
            provider, label = cfg["provider"], cfg.get("label", cfg["provider"])
            attempts = [[(provider, cfg["options"])], [provider]] if cfg["options"] else [[provider]]
            for prov_list in attempts:
                try:
                    sess = ort.InferenceSession(
                        str(self.model_path),
                        sess_options=_base_options(self.threads, _LEVELS[cfg["level"]]),
                        providers=prov_list + ["CPUExecutionProvider"])
                    if provider not in sess.get_providers():
                        raise RuntimeError(f"{label} non activé par ONNX Runtime")
                    self.session = sess
                    self.on_gpu = True
                    self.gpu_config = {k: cfg[k] for k in ("provider", "level", "options")}
                    self._gpu_label = label
                    self.device_used = f"Carte graphique ({label}) – {self._gpu_name or label}"
                    log.info("session GPU ouverte : %s, niveau %s, options %r",
                             self.device_used, cfg["level"], prov_list[0])
                    return True
                except Exception as exc:  # noqa: BLE001
                    log.info("configuration GPU refusée (%s, %r) : %s", cfg["level"], prov_list[0], exc)
        return False

    def _use_cpu(self) -> None:
        if self._cpu_session is None:
            self._cpu_session = ort.InferenceSession(
                str(self.model_path), sess_options=_base_options(self.threads),
                providers=["CPUExecutionProvider"])
            log.info("session CPU créée (%d threads)", self.threads or default_threads())
        self.session = self._cpu_session
        self.on_gpu = False
        self.gpu_config = None
        self.device_used = f"Processeur ({self.threads or default_threads()} threads)"

    # -- inférence ---------------------------------------------------------
    @staticmethod
    def _infer_raw(session: ort.InferenceSession, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, xt = session.run(["x", "xt"], dsp.model_inputs(chunk))
        return x[0, VOCALS_ROW], xt[0, VOCALS_ROW]

    @classmethod
    def _infer(cls, session: ort.InferenceSession, chunk: np.ndarray) -> np.ndarray:
        """(2, N) -> voix (2, N) : STFT numpy, modèle, iSTFT numpy."""
        x_src, xt_src = cls._infer_raw(session, chunk)
        return dsp.stem_from_outputs(x_src, xt_src, chunk.shape[-1])

    def _run_cpu(self, x: np.ndarray) -> np.ndarray:
        if self._cpu_session is None:
            self._use_cpu() if not self.on_gpu else self._ensure_cpu_session()
        return self._infer(self._cpu_session, x)

    def _ensure_cpu_session(self) -> None:
        if self._cpu_session is None:
            self._cpu_session = ort.InferenceSession(
                str(self.model_path), sess_options=_base_options(self.threads),
                providers=["CPUExecutionProvider"])
            log.info("session CPU créée (%d threads)", self.threads or default_threads())

    def _fallback(self, reason: str) -> None:
        log.warning("repli CPU : %s", reason)
        self.notes.append(reason + ", repli sur le processeur")
        _save_gpu_config(None)
        self.session = None
        self._use_cpu()

    def _run_gpu_guarded(self, x: np.ndarray, cancel: threading.Event | None,
                         timeout: float) -> tuple[np.ndarray | None, str | None]:
        """Exécute sur le GPU dans un thread surveillé. Retourne (résultat, erreur)."""
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
        ref = dsp.stem_from_outputs(ref_x, ref_xt, x.shape[-1])
        log.info("référence CPU du premier morceau : %.1f s", time.time() - t0)
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
                log.info("vérification GPU (%s, %r) : %.1f s, fini=%s, écart voix %.3g "
                         "(spectre %.3g, temporel %.3g ; rms gpu %.3g / cpu %.3g)",
                         self.gpu_config["level"], self.gpu_config["options"], dt, finite, rel,
                         rel_x, rel_xt, float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(ref ** 2))))
                if rel < self.VERIFY_TOLERANCE:
                    self.gpu_runs += 1
                    _save_gpu_config(self.gpu_config)
                    if tried > 1:
                        self.notes.append(f"carte graphique validée après {tried} configurations "
                                          f"(niveau {self.gpu_config['level']}, options {self.gpu_config['options']})")
                    return y
                err = f"résultat faux sur la carte graphique (écart {rel:.0%} vs processeur)"
            log.warning("configuration GPU écartée : %s", err)
            self.session = None
            if not self._open_next_gpu_config():
                self.on_gpu = False
                self._fallback(f"{err} ; {tried} configuration(s) essayée(s)")
                return ref
        return ref

    def run(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Voix estimée (2, N) pour un segment (2, N)."""
        if not self.on_gpu:
            return self._run_cpu(x)
        if self.gpu_runs == 0 and self.verify:
            if float(np.sqrt(np.mean(x ** 2))) < self.VERIFY_MIN_RMS:
                # Morceau quasi silencieux (intro) : comparaison peu fiable,
                # on le calcule sur le CPU et on vérifiera le prochain.
                log.info("morceau quasi silencieux : calcul CPU, vérification GPU reportée")
                return self._run_cpu(x)
            return self._verify_first_chunk(x, cancel)
        timeout = self.GPU_FIRST_TIMEOUT_S if self.gpu_runs == 0 else self.GPU_TIMEOUT_S
        t0 = time.time()
        raw, err = self._run_gpu_guarded(x, cancel, timeout)
        if raw is None:
            self._fallback(err or "erreur inconnue")
            return self._run_cpu(x)
        gx, gxt = raw
        if not (np.isfinite(gx).all() and np.isfinite(gxt).all()):
            self._fallback("résultat invalide sur la carte graphique (NaN)")
            return self._run_cpu(x)
        self.gpu_runs += 1
        log.info("morceau GPU %d : %.1f s", self.gpu_runs, time.time() - t0)
        return dsp.stem_from_outputs(gx, gxt, x.shape[-1])

    def close(self) -> None:
        self.session = None
        self._cpu_session = None


def make_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    """Compatibilité : session CPU seule."""
    return Separator(model_path, "cpu", threads).session


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
# Séparation
# --------------------------------------------------------------------------
def _transition_window(segment: int, overlap_frac: float = 0.25) -> np.ndarray:
    t = int(segment * overlap_frac)
    w = np.ones(segment, dtype=np.float32)
    fade = np.linspace(0, 1, t, dtype=np.float32)
    w[:t] = fade
    w[-t:] = fade[::-1]
    return w


def separate_vocals(sep: Separator, mix: np.ndarray, *,
                    progress: ProgressCb | None = None,
                    cancel: threading.Event | None = None) -> np.ndarray:
    """Voix estimée, (2, n) à 44,1 kHz. Découpage 7,8 s avec recouvrement 25 %."""
    total = mix.shape[1]
    overlap = N_SAMPLES // 4
    stride = N_SAMPLES - overlap
    n_chunks = max(1, (total + stride - 1) // stride)
    window = _transition_window(N_SAMPLES)
    out = np.zeros((2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)

    for i in range(n_chunks):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        start = i * stride
        end = min(start + N_SAMPLES, total)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])))
        x = np.ascontiguousarray(chunk, dtype=np.float32)
        if progress:
            extra = " (première passe, peut prendre un moment)" if i == 0 and sep.on_gpu else ""
            progress(i / n_chunks, f"Séparation… morceau {i + 1}/{n_chunks} sur {sep.device_used}{extra}")
        t_chunk = time.time()
        vocals = sep.run(x, cancel)                         # (2, N)
        if i == 0:
            log.info("premier morceau : %.1f s sur %s", time.time() - t_chunk, sep.device_used)
        n = end - start
        out[:, start:end] += vocals[:, :n] * window[:n]
        weight[start:end] += window[:n]
    return out / np.maximum(weight, 1e-8)


def output_path_for(src: Path, fmt: str = "wav") -> Path:
    return src.with_name(f"{src.stem} (instrumental).{fmt}")


def make_instrumental(src: Path, dst: Path | None = None, *, fmt: str = "wav",
                      device: Device = "auto",
                      threads: int | None = None,
                      progress: ProgressCb | None = None,
                      cancel: threading.Event | None = None,
                      info: dict | None = None) -> Path:
    """Pipeline complet : fichier audio -> fichier instrumental (sans voix).

    ``info`` (optionnel) reçoit "device" (périphérique utilisé), "notes"
    (diagnostics) et "seconds" (durée du traitement).
    """
    t_start = time.time()
    setup_logging()
    log.info("début : %s -> %s (device=%s, fmt=%s)", src, dst, device, fmt)
    src = Path(src)
    dst = Path(dst) if dst else output_path_for(src, fmt)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.0, "Préparation du modèle…")
    model = ensure_model(lambda f, m: report(0.05 * f, m), cancel)
    report(0.05, "Chargement du modèle (quelques secondes)…")
    sep = Separator(model, device, threads)
    report(0.08, f"Calcul sur : {sep.device_used}")

    report(0.10, "Lecture du fichier audio…")
    native, sr = load_audio(src)
    mix = resample(native, sr, MODEL_SAMPLE_RATE)

    vocals = separate_vocals(sep, mix,
                             progress=lambda f, m: report(0.10 + 0.85 * f, m),
                             cancel=cancel)
    if info is not None:
        info["device"] = sep.device_used
        info["notes"] = list(sep.notes)
    sep.close()

    report(0.95, "Écriture du fichier…")
    vocals_native = resample(vocals, MODEL_SAMPLE_RATE, sr)
    n = min(native.shape[1], vocals_native.shape[1])
    instrumental = native[:, :n] - vocals_native[:, :n]
    write_audio(dst, instrumental, sr)
    if info is not None:
        info["seconds"] = time.time() - t_start
    log.info("terminé en %.0f s sur %s", time.time() - t_start, sep.device_used)
    report(1.0, "Terminé")
    return dst


def speed_test(progress: ProgressCb | None = None,
               cancel: threading.Event | None = None) -> str:
    """Mesure un morceau de 7,8 s sur la carte graphique (si présente) et sur le CPU."""
    setup_logging()
    lines = [describe_devices(), ""]
    model = ensure_model(progress, cancel)
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
            sep = Separator(model, device)
            t0 = time.time()
            y = sep.run(x, cancel)              # première passe (compilation)
            first = time.time() - t0
            t0 = time.time()
            y = sep.run(x, cancel)
            second = time.time() - t0
            results[device] = y
            if sep.notes:
                lines.append(f"{label} : {'; '.join(sep.notes)}")
            else:
                lines.append(f"{label} ({sep.device_used}) : 1er morceau {first:.1f} s, "
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
    lines += ["", f"Journal : {log_path()}"]
    text = "\n".join(lines)
    log.info("test de vitesse :\n%s", text)
    return text
