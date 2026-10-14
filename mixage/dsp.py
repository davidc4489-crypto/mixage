"""STFT / iSTFT en numpy, reproduisant exactement `HTDemucs._spec` / `_ispec`
(torch.stft normalized=True, center=True, fenêtre de Hann périodique).

Le modèle « core » (export sans transformée dans le graphe) prend :
  mix  (1, 2, 343980)        forme d'onde 44,1 kHz
  mag  (1, 4, 2048, 336)     spectrogramme complexe en canaux [L.re, L.im, R.re, R.im]
et renvoie :
  x    (1, 4, 4, 2048, 336)  spectrogrammes par source (même disposition)
  xt   (1, 4, 2, 343980)     branche temporelle par source
Source s = iSTFT(x[s]) + xt[s].
"""
from __future__ import annotations

import math

import numpy as np

NFFT = 4096
HOP = NFFT // 4            # 1024
FREQS = NFFT // 2          # 2048 bins gardés (le dernier est supprimé)
_PAD = HOP // 2 * 3        # 1536, cf. HTDemucs._spec


def _hann_periodic(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


_WINDOW = _hann_periodic(NFFT)


def _reflect_pad(x: np.ndarray, left: int, right: int) -> np.ndarray:
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="reflect")


def _stft(x: np.ndarray) -> np.ndarray:
    """torch.stft(normalized=True, center=True, pad_mode=reflect) sur (..., L)."""
    x = _reflect_pad(x, NFFT // 2, NFFT // 2)
    n_frames = 1 + (x.shape[-1] - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n_frames)[:, None]   # (T, NFFT)
    frames = x[..., idx] * _WINDOW                                          # (..., T, NFFT)
    z = np.fft.rfft(frames, axis=-1) / math.sqrt(NFFT)                      # (..., T, NFFT/2+1)
    return np.swapaxes(z, -1, -2)                                           # (..., F, T)


def _istft(z: np.ndarray, length: int) -> np.ndarray:
    """torch.istft(normalized=True, center=True, length=length) sur (..., F, T)."""
    n_frames = z.shape[-1]
    frames = np.fft.irfft(np.swapaxes(z, -1, -2) * math.sqrt(NFFT), n=NFFT, axis=-1)  # (..., T, NFFT)
    frames = frames * _WINDOW
    total = NFFT + HOP * (n_frames - 1)
    out = np.zeros(z.shape[:-2] + (total,), dtype=np.float64)
    env = np.zeros(total, dtype=np.float64)
    w2 = (_WINDOW.astype(np.float64)) ** 2
    for t in range(n_frames):
        s = t * HOP
        out[..., s:s + NFFT] += frames[..., t, :]
        env[s:s + NFFT] += w2
    out = out[..., NFFT // 2: NFFT // 2 + length]
    env = env[NFFT // 2: NFFT // 2 + length]
    env = np.where(env > 1e-11, env, 1.0)
    return (out / env).astype(np.float32)


def spec(mix: np.ndarray) -> np.ndarray:
    """HTDemucs._spec : (C, L) -> complexe (C, 2048, ceil(L/HOP))."""
    length = mix.shape[-1]
    le = int(math.ceil(length / HOP))
    x = _reflect_pad(mix, _PAD, _PAD + le * HOP - length)
    z = _stft(x)[..., :-1, :]
    assert z.shape[-1] == le + 4, (z.shape, le)
    return z[..., 2:2 + le]


def ispec(z: np.ndarray, length: int) -> np.ndarray:
    """HTDemucs._ispec : complexe (..., 2048, T) -> (..., length)."""
    z = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (2, 2)])
    le = HOP * int(math.ceil(length / HOP)) + 2 * _PAD
    x = _istft(z, le)
    return x[..., _PAD:_PAD + length]


def cac_from_complex(z: np.ndarray) -> np.ndarray:
    """(C, F, T) complexe -> (2C, F, T) réel [c0.re, c0.im, c1.re, c1.im]."""
    c, f, t = z.shape
    m = np.stack([z.real, z.imag], axis=1)            # (C, 2, F, T)
    return m.reshape(c * 2, f, t).astype(np.float32)


def complex_from_cac(m: np.ndarray) -> np.ndarray:
    """(2C, F, T) réel -> (C, F, T) complexe."""
    c2, f, t = m.shape
    m = m.reshape(c2 // 2, 2, f, t)
    return m[:, 0] + 1j * m[:, 1]


def model_inputs(mix: np.ndarray) -> dict[str, np.ndarray]:
    """Prépare les deux entrées du modèle « core » pour un segment (2, 343980)."""
    z = spec(mix.astype(np.float32))
    return {"mix": mix[np.newaxis].astype(np.float32),
            "mag": cac_from_complex(z)[np.newaxis]}


def stem_from_outputs(x_src: np.ndarray, xt_src: np.ndarray, length: int) -> np.ndarray:
    """Une source : x_src (4, 2048, T) cac + xt_src (2, length) -> (2, length)."""
    return ispec(complex_from_cac(x_src), length) + xt_src
