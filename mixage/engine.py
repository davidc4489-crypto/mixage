"""Moteur de séparation : HT-Demucs (spécialiste voix) via ONNX Runtime.

Pas de PyTorch. Entrée : n'importe quel fichier audio lisible par
libsndfile (mp3, wav, flac, ogg...). Sortie : instrumental = mix - voix.

Réglages mémoire importants (mesurés sur un PC 8 cœurs / 15 Go) :
- optimisations de graphe ONNX activées + arena mémoire  -> > 7 Go, plantage
- optimisations désactivées + sans arena                  -> ~0,9 à 1,7 Go
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

MODEL_SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * MODEL_SAMPLE_RATE)  # 343 980, figé dans le graphe ONNX
SOURCES = ("drums", "bass", "other", "vocals")
VOCALS_ROW = SOURCES.index("vocals")

MODEL_FILENAME = "htdemucs_ft_vocals_fp16weights.onnx"
MODEL_URL = ("https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/"
             "resolve/main/" + MODEL_FILENAME)
MODEL_SIZE_BYTES = 165_612_636

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
    # Les optimisations de graphe (constant folding) font exploser la RAM
    # sur ce modèle ; on les désactive. L'arena pré-alloue aussi des Go.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False          # obligatoire avec DirectML
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

    # -- inférence ---------------------------------------------------------
    def run(self, x: np.ndarray) -> np.ndarray:
        """Retourne les stems (4, 2, N) ; bascule sur le CPU si le GPU échoue."""
        assert self.session is not None
        try:
            return self.session.run(["stems"], {"mix": x})[0][0]
        except Exception as exc:  # noqa: BLE001
            if not self.on_gpu:
                raise
            self.notes.append(f"échec sur la carte graphique en cours de traitement "
                              f"({type(exc).__name__}), repli sur le processeur")
            self.session = None
            self._use_cpu()
            return self.session.run(["stems"], {"mix": x})[0][0]

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
        x = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        stems = sep.run(x)                                  # (4, 2, N)
        n = end - start
        out[:, start:end] += stems[VOCALS_ROW, :, :n] * window[:n]
        weight[start:end] += window[:n]
        if progress:
            progress((i + 1) / n_chunks, f"Séparation… morceau {i + 1}/{n_chunks}")
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
    report(1.0, "Terminé")
    return dst
