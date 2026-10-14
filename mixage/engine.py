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

MODEL_FILENAME = "htdemucs_ft_vocals_safe16.onnx"
MODEL_URL = ("https://huggingface.co/monteslu/htdemucs-ft-webgpu/"
             "resolve/main/" + MODEL_FILENAME)
MODEL_SIZE_BYTES = 87_695_109

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
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mixage/0.1"})
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
    return (f"ONNX Runtime {ort.__version__}\n"
            f"Accélérateurs disponibles : {provs}\n"
            f"Cartes graphiques : {gpus}")


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n, 4))


def _base_options(threads: int | None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    # Mesuré sur le modèle core : l'arena mémoire + mem_pattern font monter la
    # RAM à 5-6 Go ; sans eux, 0,4 Go avec les optimisations complètes
    # (mem_pattern doit de toute façon être désactivé avec DirectML).
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads or default_threads()
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


def _gpu_provider_attempts(provider: str) -> list[list]:
    """Variantes d'options à essayer pour un fournisseur GPU, de la plus
    précise à la plus simple (les noms d'options varient selon la version)."""
    if provider == "DmlExecutionProvider":
        return [
            [(provider, {"performance_preference": "high_performance", "device_filter": "gpu"})],
            [(provider, {"device_id": 0})],
            [provider],
        ]
    return [[provider]]


class Separator:
    """Session ONNX avec choix du périphérique et repli automatique sur le CPU.

    ``device`` : "auto" (carte graphique si possible), "gpu" (carte graphique
    obligatoire, erreur sinon) ou "cpu".
    """

    def __init__(self, model_path: Path, device: Device = "auto",
                 threads: int | None = None) -> None:
        self.model_path = Path(model_path)
        self.threads = threads
        self.session: ort.InferenceSession | None = None
        self.device_used = ""
        self.on_gpu = False
        self.notes: list[str] = []
        self.gpu_runs = 0

        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device inconnu : {device!r}")
        if device != "cpu":
            self._try_gpu(required=(device == "gpu"))
        if self.session is None:
            self._use_cpu()

    # -- création ----------------------------------------------------------
    def _try_gpu(self, required: bool) -> None:
        providers = available_gpu_providers()
        if not providers:
            msg = "aucun accélérateur graphique disponible dans cette version"
            if required:
                raise RuntimeError("Carte graphique demandée mais " + msg + ".")
            self.notes.append(msg)
            return
        gpus = list_gpus()
        if sys.platform == "win32" and not gpus:
            msg = "aucune carte graphique physique détectée"
            if required:
                raise RuntimeError("Carte graphique demandée mais " + msg + ".")
            self.notes.append(msg)
            return
        last_err: Exception | None = None
        for provider, label in providers:
            for attempt in _gpu_provider_attempts(provider):
                try:
                    sess = ort.InferenceSession(str(self.model_path),
                                                sess_options=_base_options(self.threads),
                                                providers=attempt + ["CPUExecutionProvider"])
                    if provider not in sess.get_providers():
                        raise RuntimeError(f"{label} non activé par ONNX Runtime")
                    self.session = sess
                    self.on_gpu = True
                    name = gpus[0] if gpus else label
                    self.device_used = f"Carte graphique ({label}) – {name}"
                    log.info("session GPU créée : %s avec options %r", self.device_used, attempt)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
        msg = f"carte graphique inutilisable ({type(last_err).__name__}: {last_err})"
        if required:
            raise RuntimeError(msg)
        self.notes.append(msg)

    def _use_cpu(self) -> None:
        self.session = ort.InferenceSession(str(self.model_path),
                                            sess_options=_base_options(self.threads),
                                            providers=["CPUExecutionProvider"])
        self.on_gpu = False
        self.device_used = f"Processeur ({self.threads or default_threads()} threads)"
        log.info("session CPU créée : %s", self.device_used)

    # -- inférence ---------------------------------------------------------
    #: Délai maximal pour le premier morceau sur carte graphique (compilation
    #: des opérateurs incluse), puis pour les suivants. Au-delà : repli CPU.
    GPU_FIRST_TIMEOUT_S = 150.0
    GPU_TIMEOUT_S = 25.0

    @staticmethod
    def _infer(session: ort.InferenceSession, chunk: np.ndarray) -> np.ndarray:
        """(2, N) -> voix (2, N) : STFT numpy, modèle, iSTFT numpy."""
        x, xt = session.run(["x", "xt"], dsp.model_inputs(chunk))
        return dsp.stem_from_outputs(x[0, VOCALS_ROW], xt[0, VOCALS_ROW], chunk.shape[-1])

    def _run_cpu(self, x: np.ndarray) -> np.ndarray:
        assert self.session is not None
        return self._infer(self.session, x)

    def _fallback(self, reason: str) -> None:
        log.warning("repli CPU : %s", reason)
        self.notes.append(reason + ", repli sur le processeur")
        self.session = None
        self._use_cpu()

    def run(self, x: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Voix estimée (2, N) pour un segment (2, N).

        Sur carte graphique, le calcul tourne dans un thread surveillé : s'il
        échoue, dépasse le délai de garde ou si l'utilisateur annule, on
        bascule sur le processeur (le thread orphelin est abandonné).
        """
        assert self.session is not None
        if not self.on_gpu:
            return self._run_cpu(x)

        timeout = self.GPU_FIRST_TIMEOUT_S if self.gpu_runs == 0 else self.GPU_TIMEOUT_S
        result: dict = {}
        session = self.session

        def worker() -> None:
            try:
                result["y"] = self._infer(session, x)
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
                self._fallback(f"carte graphique trop lente (plus de {timeout:.0f} s pour un morceau)")
                return self._run_cpu(x)
        if "err" in result:
            self._fallback(f"échec sur la carte graphique ({type(result['err']).__name__}: {result['err']})")
            return self._run_cpu(x)
        dt = time.time() - t0
        self.gpu_runs += 1
        log.info("morceau GPU %d : %.1f s", self.gpu_runs, dt)
        return result["y"]

    def close(self) -> None:
        self.session = None


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
    x = (np.random.randn(2, N_SAMPLES) * 0.1).astype(np.float32)
    for device, label in (("gpu", "Carte graphique"), ("cpu", "Processeur")):
        if progress:
            progress(0.5 if device == "cpu" else 0.1, f"Test {label}…")
        try:
            sep = Separator(model, device)
            t0 = time.time()
            sep.run(x, cancel)                  # première passe (compilation)
            first = time.time() - t0
            t0 = time.time()
            sep.run(x, cancel)
            second = time.time() - t0
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
    lines += ["", f"Journal : {log_path()}"]
    text = "\n".join(lines)
    log.info("test de vitesse :\n%s", text)
    return text
