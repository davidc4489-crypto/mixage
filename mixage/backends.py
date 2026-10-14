"""Accélérateurs matériels pour ONNX Runtime : WebGPU (plugin) et DirectML.

WebGPU : plugin officiel ``onnxruntime-ep-webgpu`` (Microsoft, MIT). Sous
Windows il passe par Direct3D 12 (cartes NVIDIA, AMD, Intel) ; sous Linux par
Vulkan. C'est le même moteur que celui pour lequel le modèle « core » a été
exporté (il tourne dans le navigateur avec onnxruntime-web). Vérifié exact
(écart 4e-5 contre le CPU) sur un Intel UHD 620 via Vulkan.

DirectML : ancien fournisseur Windows (paquet ``onnxruntime-directml``) ; il
donnait un résultat faux avec ce modèle sur la carte de l'utilisateur, quelle
que soit la configuration. Il reste géré si le paquet est présent, mais il
n'est plus livré dans l'exécutable.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
import threading
from pathlib import Path

import onnxruntime as ort

log = logging.getLogger("mixage")

WEBGPU_EP = "WebGpuExecutionProvider"
DML_EP = "DmlExecutionProvider"

LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}

#: Adaptateurs logiciels Windows à ignorer : pas de vraie carte graphique.
_SOFTWARE_ADAPTERS = ("basic display", "basic render", "remote display", "warp", "virtual")


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n, 4))


def base_options(threads: int | None,
                 level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                 ) -> ort.SessionOptions:
    """Options de session communes.

    Mesuré sur le modèle core : l'arena mémoire + mem_pattern font monter la
    RAM à 5-6 Go ; sans eux, 0,4 Go au repos (pic transitoire ~2,6 Go par
    morceau). mem_pattern doit de toute façon être désactivé avec DirectML.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads or default_threads()
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


# --------------------------------------------------------------------------
# WebGPU (plugin)
# --------------------------------------------------------------------------
_webgpu_lock = threading.Lock()
_webgpu_state: dict = {}


def webgpu_status() -> tuple[list, str]:
    """Enregistre le plugin WebGPU (une seule fois) ; (périphériques, erreur)."""
    with _webgpu_lock:
        if not _webgpu_state:
            try:
                import onnxruntime_ep_webgpu as webgpu
                lib = Path(webgpu.get_library_path())
                if sys.platform == "win32":
                    # dxcompiler.dll / dxil.dll sont à côté du plugin
                    os.environ["PATH"] = str(lib.parent) + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(str(lib.parent))
                    except Exception:  # noqa: BLE001
                        pass
                ort.register_execution_provider_library("webgpu", str(lib))
                devs = [d for d in ort.get_ep_devices() if d.ep_name == WEBGPU_EP]
                _webgpu_state["devices"] = devs
                _webgpu_state["error"] = "" if devs else "aucun périphérique WebGPU (pilote graphique absent ou trop ancien ?)"
                log.info("WebGPU : plugin chargé, %d périphérique(s)", len(devs))
            except ImportError:
                _webgpu_state["devices"] = []
                _webgpu_state["error"] = "plugin onnxruntime-ep-webgpu absent"
            except Exception as exc:  # noqa: BLE001
                _webgpu_state["devices"] = []
                _webgpu_state["error"] = f"{type(exc).__name__}: {exc}"
                log.info("WebGPU indisponible : %s", _webgpu_state["error"])
        return _webgpu_state["devices"], _webgpu_state["error"]


def dml_available() -> bool:
    return DML_EP in ort.get_available_providers()


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
    except Exception:  # noqa: BLE001
        return []
    names = [n.strip() for n in out.splitlines() if n.strip()]
    return [n for n in names if not any(k in n.lower() for k in _SOFTWARE_ADAPTERS)]


# --------------------------------------------------------------------------
# Configurations candidates et ouverture de session
# --------------------------------------------------------------------------
def gpu_configs() -> list[dict]:
    """Configurations GPU à essayer, dans l'ordre (chacune est vérifiée contre le CPU).

    WebGPU d'abord (exact sur ce modèle), avec disposition NHWC (20 % plus
    rapide que NCHW, mesuré) puis NCHW sans optimisation de graphe. DirectML
    ensuite si le paquet est présent : avec puis sans « metacommands » (noyaux
    du pilote, source connue de résultats faux), à des niveaux d'optimisation
    décroissants.
    """
    cfgs: list[dict] = []
    devs, _ = webgpu_status()
    if devs:
        cfgs += [
            {"backend": "webgpu", "level": "all", "options": {"preferredLayout": "NHWC"}},
            {"backend": "webgpu", "level": "disable", "options": {"preferredLayout": "NCHW"}},
        ]
    if dml_available():
        cfgs += [
            {"backend": "dml", "level": "all", "options": {"device_id": 0}},
            {"backend": "dml", "level": "all", "options": {"device_id": 0, "disable_metacommands": "1"}},
            {"backend": "dml", "level": "basic", "options": {"device_id": 0, "disable_metacommands": "1"}},
            {"backend": "dml", "level": "disable", "options": {"device_id": 0, "disable_metacommands": "1"}},
        ]
    return cfgs


def backend_label(cfg: dict) -> str:
    return {"webgpu": "WebGPU", "dml": "DirectML"}.get(cfg["backend"], cfg["backend"])


def open_gpu_session(model_path: Path, cfg: dict, threads: int | None) -> ort.InferenceSession:
    """Ouvre une session sur l'accélérateur décrit par ``cfg`` (lève si impossible)."""
    so = base_options(threads, LEVELS[cfg["level"]])
    if cfg["backend"] == "webgpu":
        devs, err = webgpu_status()
        if not devs:
            raise RuntimeError(err or "WebGPU indisponible")
        so.add_provider_for_devices(devs[:1], dict(cfg["options"]))
        sess = ort.InferenceSession(str(model_path), sess_options=so)
        if WEBGPU_EP not in sess.get_providers():
            raise RuntimeError("WebGPU non activé par ONNX Runtime")
        return sess
    if cfg["backend"] == "dml":
        sess = ort.InferenceSession(str(model_path), sess_options=so,
                                    providers=[(DML_EP, dict(cfg["options"])), "CPUExecutionProvider"])
        if DML_EP not in sess.get_providers():
            raise RuntimeError("DirectML non activé par ONNX Runtime")
        return sess
    raise ValueError(f"accélérateur inconnu : {cfg['backend']!r}")


def open_cpu_session(model_path: Path, threads: int | None) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), sess_options=base_options(threads),
                                providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------
# Configuration validée, mémorisée entre deux lancements
# --------------------------------------------------------------------------
def load_saved_config(path: Path, model_name: str) -> dict | None:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("model") == model_name and cfg.get("ort") == ort.__version__:
            c = cfg["config"]
            if {"backend", "level", "options"} <= set(c):
                return c
    except Exception:  # noqa: BLE001
        pass
    return None


def save_config(path: Path, model_name: str, cfg: dict | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if cfg is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps({"model": model_name, "ort": ort.__version__, "config": cfg},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# Mémoire vive (pour décider si processeur et carte peuvent travailler ensemble)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def total_ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2**30
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 2**30
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def describe(saved: dict | None) -> str:
    """Texte de diagnostic : accélérateurs disponibles et cartes détectées."""
    devs, err = webgpu_status()
    webgpu = f"disponible ({len(devs)} périphérique(s))" if devs else f"indisponible – {err}"
    dml = "disponible" if dml_available() else "non inclus"
    gpus = ", ".join(list_gpus()) or ("non détectée" if sys.platform == "win32" else "n/a hors Windows")
    cfg = (f"{backend_label(saved)}, niveau {saved['level']}, options {saved['options']}" if saved
           else "aucune encore validée")
    return (f"ONNX Runtime {ort.__version__}\n"
            f"WebGPU : {webgpu}\n"
            f"DirectML : {dml}\n"
            f"Cartes graphiques : {gpus}\n"
            f"Mémoire vive : {total_ram_gb():.0f} Go\n"
            f"Configuration carte graphique mémorisée : {cfg}")
