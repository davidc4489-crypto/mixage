"""Mode rapide : Spleeter 2 stems (Deezer, MIT) en ONNX, sur le processeur.

Deux petits U-Net (10 M de paramètres chacun) travaillent sur le spectrogramme
d'amplitude (fenêtres de 4096, pas 1024, 1024 premiers bins = 0 à 11 kHz)
par segments de 512 trames (~11,9 s). Chaque réseau estime l'amplitude d'une
source ; les deux estimations sont combinées en un masque de Wiener
(exposant 2) appliqué à la STFT complexe du mix. La voix est reconstruite par
iSTFT ; l'instrumental vaut mix − voix, donc tout ce qui est au-dessus de
11 kHz reste intact dans l'instrumental.

Export ONNX : projet sherpa-onnx (k2-fsa, Apache 2.0), entrée ``x``
(2 canaux, segments, 512, 1024), sortie ``y`` de même forme. Mesuré sur un
i5 4 cœurs : 0,7 s pour 30 s d'audio (environ 45 fois plus vite que le
temps réel), corrélation 0,98 avec l'instrumental HT-Demucs.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from . import backends

log = logging.getLogger("mixage")

NFFT = 4096
HOP = 1024
F_BINS = 1024          # bins gardés par le réseau
T_FRAMES = 512         # trames par segment
BATCH_SEGMENTS = 4     # segments par passage (mémoire ~ 150 Mo)
EPS = 1e-10

_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32)  # Hann périodique

ProgressCb = Callable[[float, str], None]


#: Zéros ajoutés avant le signal : ainsi chaque échantillon réel est couvert par
#: les 4 fenêtres qui se recouvrent et la reconstruction est exacte dès le début
#: (sans cela, le premier échantillon n'est couvert que par le pied d'une fenêtre
#: et la normalisation l'amplifie énormément).
_LEAD = NFFT - HOP


def _stft(x: np.ndarray) -> tuple[np.ndarray, int]:
    """STFT façon tf.signal.stft(4096, 1024, pad_end=True), signal précédé de _LEAD zéros."""
    n = x.shape[-1] + _LEAD
    n_frames = int(math.ceil(n / HOP))
    total = (n_frames - 1) * HOP + NFFT
    xp = np.pad(x, ((0, 0), (_LEAD, total - n)))
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[:, idx] * _WINDOW, axis=-1), total


def _istft(z: np.ndarray, total: int, n: int) -> np.ndarray:
    """Inverse par recouvrement-addition (fenêtre de synthèse = fenêtre / somme des carrés)."""
    frames = np.fft.irfft(z, n=NFFT, axis=-1).astype(np.float32) * _WINDOW   # (2, T, NFFT)
    n_frames = frames.shape[1]
    out = np.zeros((2, total), dtype=np.float64)
    env = np.zeros(total, dtype=np.float64)
    w2 = _WINDOW.astype(np.float64) ** 2
    for t in range(n_frames):
        s = t * HOP
        out[:, s:s + NFFT] += frames[:, t]
        env[s:s + NFFT] += w2
    # Hann périodique, pas = N/4 : la somme des carrés vaut 1,5 partout où les
    # fenêtres se recouvrent complètement ; plancher contre les bords.
    env = np.maximum(env, 0.5)
    return (out / env)[:, _LEAD:_LEAD + n].astype(np.float32)


class Spleeter:
    """Deux sessions ONNX (voix, accompagnement) sur le processeur."""

    def __init__(self, model_paths: list[Path], threads: int | None = None) -> None:
        paths = {p.name: p for p in map(Path, model_paths)}
        self.threads = threads
        so = backends.base_options(threads)
        t0 = time.time()
        self.vocals_net = ort.InferenceSession(str(paths["spleeter_vocals.onnx"]), so,
                                               providers=["CPUExecutionProvider"])
        self.accomp_net = ort.InferenceSession(str(paths["spleeter_accompaniment.onnx"]), so,
                                               providers=["CPUExecutionProvider"])
        log.info("Spleeter chargé en %.1f s", time.time() - t0)

    @property
    def device_used(self) -> str:
        return f"Processeur ({self.threads or backends.default_threads()} threads), mode rapide Spleeter"

    def vocals(self, mix: np.ndarray, *, progress: ProgressCb | None = None,
               cancel: threading.Event | None = None) -> np.ndarray:
        """Voix estimée (2, n) pour un mix (2, n) à 44,1 kHz."""
        from .engine import Cancelled  # import tardif (dépendance circulaire)

        n = mix.shape[-1]
        if progress:
            progress(0.0, "Analyse du spectre…")
        z, total = _stft(mix.astype(np.float32))                 # (2, T, 2049)
        n_frames = z.shape[1]
        mag = np.abs(z[:, :, :F_BINS]).astype(np.float32)        # (2, T, 1024)
        pad_t = (-n_frames) % T_FRAMES
        mag = np.pad(mag, ((0, 0), (0, pad_t), (0, 0))).reshape(2, -1, T_FRAMES, F_BINS)
        n_seg = mag.shape[1]

        mask = np.empty_like(mag)
        for s in range(0, n_seg, BATCH_SEGMENTS):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if progress:
                progress(0.1 + 0.8 * s / n_seg,
                         f"Séparation rapide… segment {s + 1}/{n_seg} sur {self.device_used}")
            batch = np.ascontiguousarray(mag[:, s:s + BATCH_SEGMENTS])
            v = self.vocals_net.run(["y"], {"x": batch})[0]
            a = self.accomp_net.run(["y"], {"x": batch})[0]
            mask[:, s:s + BATCH_SEGMENTS] = (v ** 2 + EPS / 2) / (v ** 2 + a ** 2 + EPS)

        if progress:
            progress(0.92, "Reconstruction de la voix…")
        mask = mask.reshape(2, -1, F_BINS)[:, :n_frames]
        full = np.zeros_like(z)
        full[:, :, :F_BINS] = z[:, :, :F_BINS] * mask            # au-dessus de 11 kHz : pas de voix
        return _istft(full, total, n)
