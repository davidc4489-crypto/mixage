"""Interface graphique Mixage (tkinter, glisser-déposer via tkinterdnd2).

Trois étapes : la chanson, les réglages (format, qualité), le fichier de
sortie (emplacement et nom libres). Le choix processeur / carte graphique est
automatique ; les diagnostics restent en ligne de commande (``--info``,
``--speed-test``) et dans le journal.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import __version__
from .engine import Cancelled, log_path, make_instrumental, output_path_for

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:  # pragma: no cover - dépendance optionnelle
    _HAS_DND = False

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus")
FORMATS = (("WAV – qualité maximale", "wav"), ("MP3 – plus léger", "mp3"))
MODELS = (("Rapide – quelques secondes", "spleeter"),
          ("Meilleure qualité – quelques minutes", "demucs"))

# Palette
BG = "#f4f6fa"          # fond de fenêtre
CARD = "#ffffff"        # cartes
BORDER = "#d9dee7"
TEXT = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#2563eb"      # bleu
ACCENT_DARK = "#1d4ed8"
DROP_BG = "#eef3ff"
DROP_HOVER = "#dfe8ff"
OK_GREEN = "#15803d"
ERR_RED = "#b91c1c"


def _open_path(path: Path, select: bool = False) -> None:
    """Ouvre un fichier (lecteur par défaut) ou son dossier dans l'explorateur."""
    try:
        if sys.platform == "win32":
            if select:
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)] if select else ["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent if select else path)])
    except Exception:  # noqa: BLE001
        pass


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"


class App:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
        self.root.title("Mixage – Retirer la voix d'une chanson")
        self.root.configure(bg=BG)
        self.root.geometry("620x640")
        self.root.minsize(560, 600)

        self.src: Path | None = None
        self.result: Path | None = None
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.events: queue.Queue = queue.Queue()
        self.t_start = 0.0
        self.out_edited = False   # l'utilisateur a saisi lui-même le chemin de sortie

        self.fmt = tk.StringVar(value="wav")
        self.model = tk.StringVar(value="spleeter")
        self.out_path = tk.StringVar(value="")
        self.fmt.trace_add("write", lambda *_: self._on_format_change())

        self._fonts()
        self._styles()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ look
    def _fonts(self) -> None:
        family = "Segoe UI" if sys.platform == "win32" else tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_title = (family, 20, "bold")
        self.f_sub = (family, 10)
        self.f_step = (family, 11, "bold")
        self.f_body = (family, 10)
        self.f_small = (family, 8)
        self.f_drop = (family, 12)
        self.f_button = (family, 11, "bold")

    def _styles(self) -> None:
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=TEXT, font=self.f_body)
        s.configure("Card.TFrame", background=CARD)
        s.configure("Card.TLabel", background=CARD, foreground=TEXT)
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED)
        s.configure("Step.TLabel", background=CARD, foreground=ACCENT, font=self.f_step)
        s.configure("Card.TRadiobutton", background=CARD, foreground=TEXT, font=self.f_body)
        s.map("Card.TRadiobutton", background=[("active", CARD)])
        s.configure("Card.TEntry", fieldbackground="#ffffff", padding=4)
        s.configure("Secondary.TButton", padding=(10, 5))
        s.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                    font=self.f_button, padding=(16, 10), borderwidth=0)
        s.map("Primary.TButton",
              background=[("disabled", "#9db4e8"), ("pressed", ACCENT_DARK), ("active", ACCENT_DARK)],
              foreground=[("disabled", "#ffffff")])
        s.configure("Blue.Horizontal.TProgressbar", troughcolor="#e5e7eb", background=ACCENT,
                    bordercolor="#e5e7eb", lightcolor=ACCENT, darkcolor=ACCENT, thickness=14)

    def _card(self, parent: tk.Misc, step: str, title: str) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="x", pady=(0, 10))
        card = ttk.Frame(outer, style="Card.TFrame", padding=(14, 10, 14, 12))
        card.pack(fill="x")
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 6))
        ttk.Label(head, text=step, style="Step.TLabel").pack(side="left")
        ttk.Label(head, text=title, style="Card.TLabel", font=self.f_step).pack(side="left", padx=(6, 0))
        return card

    # ----------------------------------------------------------------- build
    def _build(self) -> None:
        r = self.root
        body = tk.Frame(r, bg=BG, padx=22, pady=16)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Mixage", bg=BG, fg=TEXT, font=self.f_title).pack(anchor="w")
        tk.Label(body, text="Déposez une chanson, récupérez la musique sans la voix.",
                 bg=BG, fg=MUTED, font=self.f_sub).pack(anchor="w", pady=(0, 12))

        # --- 1. chanson
        c1 = self._card(body, "1", "Votre chanson")
        self.drop = tk.Label(
            c1, text="♪  Glissez votre chanson ici\nou cliquez pour la choisir",
            bg=DROP_BG, fg=ACCENT_DARK, font=self.f_drop, height=4, cursor="hand2",
            relief="flat", bd=0, highlightthickness=2, highlightbackground="#b7c7f0",
        )
        self.drop.pack(fill="x")
        self.drop.bind("<Button-1>", lambda e: self.choose_file())
        self.drop.bind("<Enter>", lambda e: self.drop.config(bg=DROP_HOVER))
        self.drop.bind("<Leave>", lambda e: self.drop.config(bg=DROP_BG))
        if _HAS_DND:
            try:
                self.drop.drop_target_register(DND_FILES)
                self.drop.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:  # noqa: BLE001
                pass
        self.file_lbl = ttk.Label(c1, text="Formats acceptés : mp3, wav, flac, ogg, m4a…", style="Muted.TLabel")
        self.file_lbl.pack(anchor="w", pady=(6, 0))

        # --- 2. réglages
        c2 = self._card(body, "2", "Réglages")
        row = ttk.Frame(c2, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Format :", style="Card.TLabel", width=10).grid(row=0, column=0, sticky="w", pady=2)
        for i, (lbl, code) in enumerate(FORMATS):
            ttk.Radiobutton(row, text=lbl, variable=self.fmt, value=code,
                            style="Card.TRadiobutton").grid(row=0, column=1 + i, sticky="w", padx=(0, 16))
        ttk.Label(row, text="Qualité :", style="Card.TLabel", width=10).grid(row=1, column=0, sticky="w", pady=2)
        for i, (lbl, code) in enumerate(MODELS):
            ttk.Radiobutton(row, text=lbl, variable=self.model, value=code,
                            style="Card.TRadiobutton").grid(row=1, column=1 + i, sticky="w", padx=(0, 16))

        # --- 3. sortie
        c3 = self._card(body, "3", "Fichier à créer")
        row3 = ttk.Frame(c3, style="Card.TFrame")
        row3.pack(fill="x")
        self.out_entry = ttk.Entry(row3, textvariable=self.out_path, style="Card.TEntry")
        self.out_entry.pack(side="left", fill="x", expand=True)
        self.out_entry.bind("<KeyRelease>", lambda e: setattr(self, "out_edited", True))
        ttk.Button(row3, text="Parcourir…", command=self.choose_output,
                   style="Secondary.TButton").pack(side="left", padx=(8, 0))
        ttk.Label(c3, text="Par défaut : même dossier que la chanson, nom suivi de « (instrumental) ».",
                  style="Muted.TLabel", font=self.f_small).pack(anchor="w", pady=(6, 0))

        # --- action
        act = tk.Frame(body, bg=BG)
        act.pack(fill="x", pady=(4, 0))
        self.start_btn = ttk.Button(act, text="Retirer la voix", command=self.start,
                                    style="Primary.TButton", state="disabled")
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(act, text="Annuler", command=self.do_cancel,
                                     style="Secondary.TButton")
        self.open_file_btn = ttk.Button(act, text="Écouter", command=lambda: self._open_result(False),
                                        style="Secondary.TButton")
        self.open_dir_btn = ttk.Button(act, text="Ouvrir le dossier", command=lambda: self._open_result(True),
                                       style="Secondary.TButton")

        self.progress = ttk.Progressbar(body, mode="determinate", maximum=1000,
                                        style="Blue.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(14, 6))
        self.status = tk.Label(body, text="Choisissez une chanson pour commencer.", bg=BG, fg=MUTED,
                               font=self.f_body, anchor="w", justify="left", wraplength=560)
        self.status.pack(fill="x")

        tk.Label(body, text=f"Tout se passe sur votre ordinateur, rien n'est envoyé en ligne.   ·   "
                            f"Mixage {__version__}",
                 bg=BG, fg=MUTED, font=self.f_small).pack(side="bottom", anchor="w", pady=(10, 0))

    # --------------------------------------------------------------- actions
    def _on_drop(self, event) -> None:
        files = self.root.tk.splitlist(event.data)
        if files:
            self.set_file(Path(files[0]))

    def choose_file(self) -> None:
        if self._busy():
            return
        f = filedialog.askopenfilename(
            title="Choisir une chanson",
            filetypes=[("Fichiers audio", " ".join(f"*{e}" for e in AUDIO_EXTS)), ("Tous les fichiers", "*.*")],
        )
        if f:
            self.set_file(Path(f))

    def set_file(self, path: Path) -> None:
        if self._busy():
            return
        if not path.is_file():
            messagebox.showerror("Mixage", f"Fichier introuvable :\n{path}")
            return
        if path.suffix.lower() not in AUDIO_EXTS:
            if not messagebox.askyesno("Mixage", "Ce fichier n'a pas une extension audio connue.\nEssayer quand même ?"):
                return
        self.src = path
        self.result = None
        self.out_edited = False
        self.out_path.set(str(output_path_for(path, self.fmt.get())))
        self.drop.config(text=f"♪  {path.name}", bg=DROP_BG)
        self.file_lbl.config(text=f"Dossier : {path.parent}")
        self.status.config(text="Prêt. Cliquez sur « Retirer la voix ».", fg=MUTED)
        self.progress["value"] = 0
        self.start_btn.config(state="normal")
        self._show_result_buttons(False)

    def _on_format_change(self) -> None:
        """Adapte l'extension du fichier de sortie au format choisi."""
        current = self.out_path.get().strip()
        if not current:
            return
        p = Path(current)
        if p.suffix.lower() in (".wav", ".mp3") and p.suffix.lower() != "." + self.fmt.get():
            self.out_path.set(str(p.with_suffix("." + self.fmt.get())))

    def choose_output(self) -> None:
        if self._busy():
            return
        current = Path(self.out_path.get()) if self.out_path.get().strip() else None
        fmt = self.fmt.get()
        f = filedialog.asksaveasfilename(
            title="Enregistrer la musique sans la voix sous…",
            initialdir=str(current.parent if current else (self.src.parent if self.src else Path.home())),
            initialfile=current.name if current else f"instrumental.{fmt}",
            defaultextension="." + fmt,
            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")] if fmt == "wav" else [("MP3", "*.mp3"), ("WAV", "*.wav")],
        )
        if f:
            p = Path(f)
            if p.suffix.lower() in (".wav", ".mp3") and p.suffix.lower() != "." + fmt:
                # l'utilisateur a choisi l'autre format dans la boîte de dialogue
                self.fmt.set(p.suffix.lower()[1:])
            self.out_path.set(str(p))
            self.out_edited = True

    def _target(self) -> Path | None:
        """Chemin de sortie validé (extension forcée au format choisi)."""
        text = self.out_path.get().strip()
        if not text:
            if not self.src:
                return None
            return output_path_for(self.src, self.fmt.get())
        p = Path(text)
        if not p.is_absolute() and self.src:
            p = self.src.parent / p
        if p.suffix.lower() != "." + self.fmt.get():
            p = p.with_suffix("." + self.fmt.get()) if p.suffix.lower() in (".wav", ".mp3") \
                else p.with_name(p.name + "." + self.fmt.get())
        return p

    def start(self) -> None:
        if not self.src or self._busy():
            return
        dst = self._target()
        if dst is None:
            return
        if self.src.resolve() == dst.resolve():
            messagebox.showerror("Mixage", "Le fichier à créer ne peut pas être la chanson d'origine.")
            return
        if not dst.parent.is_dir():
            messagebox.showerror("Mixage", f"Le dossier n'existe pas :\n{dst.parent}")
            return
        if dst.exists() and not messagebox.askyesno(
                "Mixage", f"Le fichier existe déjà :\n{dst.name}\n\nLe remplacer ?"):
            return
        self.out_path.set(str(dst))
        self.cancel.clear()
        self.t_start = time.time()
        self._set_busy(True)
        self.progress["value"] = 0
        self.status.config(text="Préparation…", fg=TEXT)
        self.worker = threading.Thread(target=self._work, args=(self.src, dst, self.model.get(), self.fmt.get()),
                                       daemon=True)
        self.worker.start()

    def _work(self, src: Path, dst: Path, model: str, fmt: str) -> None:
        info: dict = {}
        try:
            out = make_instrumental(
                src, dst, fmt=fmt, model=model, device="auto",
                progress=lambda f, m: self.events.put(("progress", f, m)),
                cancel=self.cancel, info=info,
            )
            self.events.put(("done", out, info))
        except Cancelled:
            self.events.put(("cancelled",))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def do_cancel(self) -> None:
        self.cancel.set()
        self.status.config(text="Annulation…", fg=MUTED)

    def _open_result(self, folder: bool) -> None:
        if self.result:
            _open_path(self.result, select=folder)

    # --------------------------------------------------------------- helpers
    def _busy(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.config(state="disabled" if busy or not self.src else "normal")
        if busy:
            self.cancel_btn.pack(side="left", padx=(8, 0))
            self._show_result_buttons(False)
        else:
            self.cancel_btn.pack_forget()
        self.drop.config(cursor="arrow" if busy else "hand2")
        self.out_entry.config(state="disabled" if busy else "normal")

    def _show_result_buttons(self, show: bool) -> None:
        if show:
            self.open_file_btn.pack(side="left", padx=(8, 0))
            self.open_dir_btn.pack(side="left", padx=(8, 0))
        else:
            self.open_file_btn.pack_forget()
            self.open_dir_btn.pack_forget()

    def _friendly_progress(self, frac: float, msg: str) -> str:
        """Message simple avec pourcentage et temps restant estimé."""
        if "Téléchargement" in msg:
            return msg
        if frac < 0.10:
            return "Préparation du modèle…"
        if frac >= 0.95:
            return "Écriture du fichier…"
        elapsed = time.time() - self.t_start
        done = (frac - 0.10) / 0.85
        eta = ""
        if done > 0.08 and elapsed > 3:
            remaining = elapsed * (1 - done) / done
            eta = f" – environ {_fmt_duration(remaining)} restantes"
        return f"Séparation de la voix… {done * 100:.0f} %{eta}"

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "progress":
                    self.progress["value"] = int(ev[1] * 1000)
                    self.status.config(text=self._friendly_progress(ev[1], ev[2]), fg=TEXT)
                elif kind == "done":
                    self.result = ev[1]
                    info = ev[2]
                    self.progress["value"] = 1000
                    note = ""
                    if any("repli" in n or "inutilisable" in n for n in info.get("notes", [])):
                        note = "  (calcul effectué sur le processeur)"
                    self.status.config(
                        text=f"✔ Terminé en {_fmt_duration(info.get('seconds', 0))}{note}\n{self.result}",
                        fg=OK_GREEN)
                    self._set_busy(False)
                    self._show_result_buttons(True)
                elif kind == "cancelled":
                    self.progress["value"] = 0
                    self.status.config(text="Annulé.", fg=MUTED)
                    self._set_busy(False)
                elif kind == "error":
                    self.progress["value"] = 0
                    self.status.config(text="Une erreur est survenue. Détails dans le journal :\n"
                                            f"{log_path()}", fg=ERR_RED)
                    self._set_busy(False)
                    messagebox.showerror("Mixage – erreur", ev[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _on_close(self) -> None:
        """Fermeture : annule un traitement en cours et attend un peu qu'il s'arrête."""
        if self._busy():
            self.cancel.set()
            self.status.config(text="Fermeture…", fg=MUTED)
            self.root.update_idletasks()
            self.worker.join(3.0)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        if self._busy():
            # Un calcul natif (ONNX Runtime) ne peut pas être interrompu : on quitte
            # sans attendre, sinon la fin du processus plante pendant son démontage.
            os._exit(0)


def run() -> None:
    App().run()
