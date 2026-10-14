"""`python -m mixage` : interface graphique, ou ligne de commande si un fichier est donné."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in ("--gui",):
        import argparse
        from pathlib import Path
        from .engine import DEVICES, MODELS, describe_devices, make_instrumental, speed_test

        if argv[0] == "--info":
            print(describe_devices())
            return 0
        if argv[0] == "--speed-test":
            print(speed_test(progress=lambda f, m: print(f"[{f * 100:3.0f}%] {m}", flush=True)))
            return 0

        p = argparse.ArgumentParser(prog="mixage",
                                    description="Retire la voix d'une chanson.")
        p.add_argument("input", help="fichier audio (mp3, wav, flac, ogg…)")
        p.add_argument("-o", "--output", help="fichier de sortie (défaut : à côté de l'entrée)")
        p.add_argument("--mp3", action="store_true", help="écrire un MP3 au lieu d'un WAV")
        p.add_argument("--model", choices=sorted(MODELS), default="demucs",
                       help="demucs (qualité, défaut) ou spleeter (rapide, processeur)")
        p.add_argument("--threads", type=int, default=None, help="threads CPU (défaut : 4 max)")
        p.add_argument("--device", choices=DEVICES, default="auto",
                       help="auto (carte graphique si possible, processeur en renfort si assez de "
                            "mémoire), both (carte + processeur), gpu (carte seule) ou cpu")
        a = p.parse_args(argv)

        def show(frac: float, msg: str) -> None:
            print(f"[{frac * 100:3.0f}%] {msg}", flush=True)

        info: dict = {}
        out = make_instrumental(Path(a.input), Path(a.output) if a.output else None,
                                fmt="mp3" if a.mp3 else "wav", model=a.model, device=a.device,
                                threads=a.threads, progress=show, info=info)
        for note in info.get("notes", []):
            print(f"Note : {note}")
        print(f"Calcul sur : {info.get('device', '?')} – {info.get('seconds', 0):.0f} s")
        print(f"Fichier écrit : {out}")
        return 0

    from .app import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
