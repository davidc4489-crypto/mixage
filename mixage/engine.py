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
# Session ONNX (réglages basse mémoire)
# --------------------------------------------------------------------------
def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n, 4))


def make_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    # Les optimisations de graphe (constant folding) font exploser la RAM
    # sur ce modèle ; on les désactive. L'arena pré-alloue aussi des Go.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = threads or default_threads()
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


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


def separate_vocals(session: ort.InferenceSession, mix: np.ndarray, *,
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
        stems = session.run(["stems"], {"mix": x})[0][0]   # (4, 2, N)
        n = end - start
        out[:, start:end] += stems[VOCALS_ROW, :, :n] * window[:n]
        weight[start:end] += window[:n]
        if progress:
            progress((i + 1) / n_chunks, f"Séparation… morceau {i + 1}/{n_chunks}")
    return out / np.maximum(weight, 1e-8)


def output_path_for(src: Path, fmt: str = "wav") -> Path:
    return src.with_name(f"{src.stem} (instrumental).{fmt}")


def make_instrumental(src: Path, dst: Path | None = None, *, fmt: str = "wav",
                      threads: int | None = None,
                      progress: ProgressCb | None = None,
                      cancel: threading.Event | None = None) -> Path:
    """Pipeline complet : fichier audio -> fichier instrumental (sans voix)."""
    src = Path(src)
    dst = Path(dst) if dst else output_path_for(src, fmt)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.0, "Préparation du modèle…")
    model = ensure_model(lambda f, m: report(0.05 * f, m), cancel)
    report(0.05, "Chargement du modèle (quelques secondes)…")
    session = make_session(model, threads)

    report(0.10, "Lecture du fichier audio…")
    native, sr = load_audio(src)
    mix = resample(native, sr, MODEL_SAMPLE_RATE)

    vocals = separate_vocals(session, mix,
                             progress=lambda f, m: report(0.10 + 0.85 * f, m),
                             cancel=cancel)
    del session

    report(0.95, "Écriture du fichier…")
    vocals_native = resample(vocals, MODEL_SAMPLE_RATE, sr)
    n = min(native.shape[1], vocals_native.shape[1])
    instrumental = native[:, :n] - vocals_native[:, :n]
    write_audio(dst, instrumental, sr)
    report(1.0, "Terminé")
    return dst
