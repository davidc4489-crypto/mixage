# -*- mode: python ; coding: utf-8 -*-
# Build : pyinstaller mixage.spec   (les modèles doivent être dans ./models/)
from pathlib import Path

block_cipher = None
MODEL_FILES = ["htdemucs_ft_vocals_fp32.onnx", "spleeter_vocals.onnx", "spleeter_accompaniment.onnx"]
datas = [(str(Path("models") / f), "models") for f in MODEL_FILES if (Path("models") / f).is_file()]
binaries = []

try:
    import tkinterdnd2
    tkdnd_dir = Path(tkinterdnd2.__file__).parent / "tkdnd"
    datas.append((str(tkdnd_dir), "tkinterdnd2/tkdnd"))
except Exception:
    pass

# Plugin WebGPU d'ONNX Runtime : la bibliothèque (et dxcompiler/dxil sous Windows)
# doit rester à côté de onnxruntime_ep_webgpu/__init__.py, qui la localise ;
# les DLL DirectX sont aussi copiées à la racine pour la recherche de DLL Windows.
try:
    import onnxruntime_ep_webgpu
    plugin_dir = Path(onnxruntime_ep_webgpu.__file__).parent
    for f in plugin_dir.iterdir():
        if f.suffix.lower() in (".dll", ".so", ".dylib"):
            binaries.append((str(f), "onnxruntime_ep_webgpu"))
            if f.name.lower() in ("dxcompiler.dll", "dxil.dll"):
                binaries.append((str(f), "."))
except Exception:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=["soundfile", "soxr", "lameenc", "tkinterdnd2", "onnxruntime_ep_webgpu"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "scipy", "matplotlib", "PIL", "IPython", "pytest"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    [],
    name="Mixage",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
