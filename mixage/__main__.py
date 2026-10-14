"""`python -m mixage` : interface graphique, ou ligne de commande si un fichier est donné."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in ("--gui",):
        import argparse
        from pathlib import Path
        from .engine import make_instrumental

        p = argparse.ArgumentParser(prog="mixage",
                                    description="Retire la voix d'une chanson.")
        p.add_argument("input", help="fichier audio (mp3, wav, flac, ogg…)")
        p.add_argument("-o", "--output", help="fichier de sortie (défaut : à côté de l'entrée)")
        p.add_argument("--mp3", action="store_true", help="écrire un MP3 au lieu d'un WAV")
        p.add_argument("--threads", type=int, default=None, help="threads CPU (défaut : 4 max)")
        a = p.parse_args(argv)

        def show(frac: float, msg: str) -> None:
            print(f"[{frac * 100:3.0f}%] {msg}", flush=True)

        out = make_instrumental(Path(a.input), Path(a.output) if a.output else None,
                                fmt="mp3" if a.mp3 else "wav", threads=a.threads,
                                progress=show)
        print(f"Fichier écrit : {out}")
        return 0

    from .app import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
