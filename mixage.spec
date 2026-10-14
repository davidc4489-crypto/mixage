# -*- mode: python ; coding: utf-8 -*-
# Build : pyinstaller mixage.spec   (le modèle doit être dans ./models/)
import os
from pathlib import Path

block_cipher = None
model = Path("models") / "htdemucs_ft_vocals_safe16.onnx"
datas = [(str(model), "models")] if model.is_file() else []

try:
    import tkinterdnd2
    tkdnd_dir = Path(tkinterdnd2.__file__).parent / "tkdnd"
    datas.append((str(tkdnd_dir), "tkinterdnd2/tkdnd"))
except Exception:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=["soundfile", "soxr", "lameenc", "tkinterdnd2"],
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
