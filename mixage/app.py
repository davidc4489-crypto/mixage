"""Interface graphique Mixage (tkinter, glisser-déposer via tkinterdnd2)."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .engine import Cancelled, describe_devices, log_path, make_instrumental, output_path_for, speed_test

DEVICE_CHOICES = [("Automatique (carte graphique si possible)", "auto"),
                  ("Carte graphique + processeur ensemble", "both"),
                  ("Carte graphique seule", "gpu"),
                  ("Processeur seul", "cpu")]
MODEL_CHOICES = [("Qualité – HT-Demucs (quelques minutes sur processeur)", "demucs"),
                 ("Rapide – Spleeter (quelques secondes, qualité moindre)", "spleeter")]

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:  # pragma: no cover - dépendance optionnelle
    _HAS_DND = False

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus")


def _open_folder(path: Path) -> None:
    folder = str(path.parent)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception:
        pass


class App:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
        self.root.title(f"Mixage {__version__} – Retirer la voix d'une chanson")
        self.root.geometry("640x500")
        self.root.minsize(560, 450)
        self.root.configure(padx=18, pady=16)

        self.src: Path | None = None
        self.result: Path | None = None
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.events: queue.Queue = queue.Queue()

        self.fmt = tk.StringVar(value="wav")
        self.device_label = tk.StringVar(value=DEVICE_CHOICES[0][0])
        self.model_label = tk.StringVar(value=MODEL_CHOICES[0][0])
        self._build()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        r = self.root
        style = ttk.Style(r)
        try:
            style.theme_use("vista" if sys.platform == "win32" else "clam")
        except tk.TclError:
            pass

        ttk.Label(r, text="Déposez une chanson, récupérez la musique sans la voix.",
                  font=("", 11, "bold")).pack(anchor="w")

        self.drop = tk.Label(
            r, text="Glissez un fichier audio ici\n(mp3, wav, flac, ogg…)\n\nou cliquez pour choisir",
            relief="groove", bd=2, bg="#f3f4f6", fg="#374151", height=6, cursor="hand2",
            font=("", 10),
        )
        self.drop.pack(fill="x", pady=(12, 8))
        self.drop.bind("<Button-1>", lambda e: self.choose_file())
        if _HAS_DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)

        self.file_lbl = ttk.Label(r, text="Aucun fichier sélectionné.", foreground="#6b7280")
        self.file_lbl.pack(anchor="w")

        opts = ttk.Frame(r)
        opts.pack(fill="x", pady=(10, 4))
        ttk.Label(opts, text="Format de sortie :").pack(side="left")
        ttk.Radiobutton(opts, text="WAV (qualité maximale)", variable=self.fmt, value="wav").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(opts, text="MP3 (plus léger)", variable=self.fmt, value="mp3").pack(side="left", padx=(8, 0))

        mdl = ttk.Frame(r)
        mdl.pack(fill="x", pady=(4, 0))
        ttk.Label(mdl, text="Modèle :").pack(side="left")
        ttk.Combobox(mdl, textvariable=self.model_label, state="readonly", width=52,
                     values=[lbl for lbl, _ in MODEL_CHOICES]).pack(side="left", padx=(8, 0))

        dev = ttk.Frame(r)
        dev.pack(fill="x", pady=(4, 0))
        ttk.Label(dev, text="Calcul :").pack(side="left")
        ttk.Combobox(dev, textvariable=self.device_label, state="readonly", width=34,
                     values=[lbl for lbl, _ in DEVICE_CHOICES]).pack(side="left", padx=(8, 0))
        ttk.Button(dev, text="Test de vitesse", command=self.run_speed_test, width=15).pack(side="right")
        ttk.Button(dev, text="Infos matériel", command=self.show_info, width=14).pack(side="right", padx=(0, 6))

        self.progress = ttk.Progressbar(r, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=(14, 4))
        self.status = ttk.Label(r, text="Prêt.", foreground="#374151")
        self.status.pack(anchor="w")

        btns = ttk.Frame(r)
        btns.pack(fill="x", pady=(16, 0))
        self.start_btn = ttk.Button(btns, text="Retirer la voix", command=self.start, state="disabled")
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Annuler", command=self.do_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.open_btn = ttk.Button(btns, text="Ouvrir le dossier", command=self.open_result, state="disabled")
        self.open_btn.pack(side="right")

        ttk.Label(r, text="Tout se passe sur votre ordinateur, rien n'est envoyé en ligne "
                          "(sauf le téléchargement du modèle au premier lancement si besoin).",
                  foreground="#6b7280", wraplength=580, font=("", 8)).pack(anchor="w", pady=(14, 0))

    # -------------------------------------------------------------- actions
    def _on_drop(self, event) -> None:
        files = self.root.tk.splitlist(event.data)
        if files:
            self.set_file(Path(files[0]))

    def choose_file(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        f = filedialog.askopenfilename(
            title="Choisir une chanson",
            filetypes=[("Fichiers audio", " ".join(f"*{e}" for e in AUDIO_EXTS)), ("Tous les fichiers", "*.*")],
        )
        if f:
            self.set_file(Path(f))

    def set_file(self, path: Path) -> None:
        if not path.is_file():
            messagebox.showerror("Mixage", f"Fichier introuvable :\n{path}")
            return
        if path.suffix.lower() not in AUDIO_EXTS:
            if not messagebox.askyesno("Mixage", "Ce fichier n'a pas une extension audio connue.\nEssayer quand même ?"):
                return
        self.src = path
        self.result = None
        self.file_lbl.config(text=f"Fichier : {path.name}", foreground="#111827")
        self.status.config(text="Prêt.")
        self.progress["value"] = 0
        self.start_btn.config(state="normal")
        self.open_btn.config(state="disabled")

    def device(self) -> str:
        for lbl, code in DEVICE_CHOICES:
            if lbl == self.device_label.get():
                return code
        return "auto"

    def model(self) -> str:
        for lbl, code in MODEL_CHOICES:
            if lbl == self.model_label.get():
                return code
        return "demucs"

    def show_info(self) -> None:
        messagebox.showinfo("Mixage – matériel", describe_devices() + f"\n\nJournal : {log_path()}")

    def run_speed_test(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.cancel.clear()
        self._set_busy(True)
        self.status.config(text="Test de vitesse en cours (jusqu'à quelques minutes)…")

        def work() -> None:
            try:
                self.events.put(("speed", speed_test(
                    progress=lambda f, m: self.events.put(("progress", f, m)), cancel=self.cancel)))
            except Cancelled:
                self.events.put(("cancelled",))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def start(self) -> None:
        if not self.src:
            return
        dst = output_path_for(self.src, self.fmt.get())
        if dst.exists() and not messagebox.askyesno(
                "Mixage", f"Le fichier existe déjà :\n{dst.name}\n\nLe remplacer ?"):
            return
        self.cancel.clear()
        self._set_busy(True)
        self.worker = threading.Thread(target=self._work,
                                       args=(self.src, dst, self.model(), self.device()), daemon=True)
        self.worker.start()

    def _work(self, src: Path, dst: Path, model: str, device: str) -> None:
        info: dict = {}
        try:
            out = make_instrumental(
                src, dst, fmt=self.fmt.get(), model=model, device=device,
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
        self.status.config(text="Annulation…")

    def open_result(self) -> None:
        if self.result:
            _open_folder(self.result)

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.config(state="disabled" if busy or not self.src else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        self.drop.config(state="disabled" if busy else "normal")

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "progress":
                    self.progress["value"] = int(ev[1] * 1000)
                    self.status.config(text=ev[2])
                elif kind == "done":
                    self.result = ev[1]
                    info = ev[2]
                    self.progress["value"] = 1000
                    secs = info.get("seconds", 0)
                    self.status.config(
                        text=f"Terminé en {secs // 60:.0f} min {secs % 60:02.0f} s sur "
                             f"{info.get('device', '?')} : {self.result.name}")
                    important = [n for n in info.get("notes", [])
                                 if "repli" in n or "inutilisable" in n]
                    if important:
                        messagebox.showwarning("Mixage – remarque", "\n".join(important))
                    self.open_btn.config(state="normal")
                    self._set_busy(False)
                elif kind == "speed":
                    self.progress["value"] = 0
                    self.status.config(text="Test de vitesse terminé.")
                    self._set_busy(False)
                    if self.src:
                        self.start_btn.config(state="normal")
                    messagebox.showinfo("Mixage – test de vitesse", ev[1])
                elif kind == "cancelled":
                    self.progress["value"] = 0
                    self.status.config(text="Annulé.")
                    self._set_busy(False)
                elif kind == "error":
                    self.progress["value"] = 0
                    self.status.config(text="Erreur.")
                    self._set_busy(False)
                    messagebox.showerror("Mixage – erreur", ev[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self) -> None:
        self.root.mainloop()


def run() -> None:
    App().run()
