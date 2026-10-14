# Mixage

Application de bureau (Windows) qui retire la voix d'une chanson et garde
uniquement la musique. Tout tourne en local, sans envoyer le fichier en ligne.

- Modèle : HT-Demucs (spécialiste voix, `htdemucs_ft_vocals`) au format ONNX,
  exécuté par ONNX Runtime. Pas de PyTorch : l'exécutable reste léger.
- Export « core » (monteslu/htdemucs-ft-webgpu, MIT) : la transformée de Fourier est
  calculée en numpy (`mixage/dsp.py`) et non dans le graphe ONNX. Le graphe passe de
  25 000 à 3 300 opérations, ce qui le rend exploitable sur carte graphique.
- Le modèle publié calcule en fp16 ; `tools/convert_fp32.py` le convertit en fp32 complet.
  En fp16, la variance du prologue déborde sur carte graphique (sortie NaN, fichier
  silencieux) ; en fp32 le résultat est exact et le processeur va 30 % plus vite.
- Instrumental = mix original − voix estimée (même taux d'échantillonnage que l'entrée).
- Entrées acceptées : mp3, wav, flac, ogg… Sortie : WAV ou MP3, à côté du fichier d'origine
  (`<nom> (instrumental).wav`).

## Utilisation (développement)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m mixage                 # interface graphique
.venv/bin/python -m mixage chanson.mp3     # ligne de commande (WAV)
.venv/bin/python -m mixage chanson.mp3 --mp3
```

Au premier lancement, le modèle fp32 (175 Mo, Release `model-v1` de ce dépôt) est
téléchargé dans le dossier utilisateur (`%LOCALAPPDATA%\Mixage\models` sous Windows, `~/.cache/Mixage/models` sous Linux),
sauf s'il est déjà fourni dans `./models/` ou intégré à l'exécutable.

## Exécutable Windows

Le workflow GitHub Actions `.github/workflows/build-windows.yml` construit `Mixage.exe`
(PyInstaller, un seul fichier, modèle intégré) à chaque push sur `main` et publie
une Release à chaque tag `v*`. Pour construire à la main sous Windows :

```bat
pip install -r requirements.txt pyinstaller
mkdir models
curl -L -o models\htdemucs_ft_vocals_safe16.onnx https://huggingface.co/monteslu/htdemucs-ft-webgpu/resolve/main/htdemucs_ft_vocals_safe16.onnx
pip install onnx
python tools\convert_fp32.py models\htdemucs_ft_vocals_safe16.onnx models\htdemucs_ft_vocals_fp32.onnx
pyinstaller mixage.spec
```

Le résultat est dans `dist\Mixage.exe`. Windows SmartScreen affichera un avertissement
car l'exécutable n'est pas signé : « Informations complémentaires » puis « Exécuter quand même ».

## Carte graphique (DirectML)

Sous Windows, l'application utilise la carte graphique via **DirectML** (DirectX 12),
ce qui couvre les cartes NVIDIA, AMD et Intel (intégrées ou dédiées) sans pilote
supplémentaire. Le choix se fait dans l'interface (« Calcul : Automatique / Carte graphique /
Processeur ») ou en ligne de commande avec `--device auto|gpu|cpu`. En mode automatique,
si aucune carte n'est détectée ou si elle échoue en cours de route, le traitement bascule
sur le processeur. `Mixage.exe --info` affiche les accélérateurs et cartes détectés.

## Performances et mémoire

Mesuré sur un portable i5 4 cœurs sans GPU : environ 0,8 × la durée de la chanson
(30 s d'audio en 25 s), pic mémoire d'environ 2,6 Go pendant un morceau.
Sur carte graphique via DirectML, la vitesse dépend de la carte (à mesurer avec le
bouton « Test de vitesse » de l'interface, qui chronomètre un morceau de 7,8 s sur la
carte puis sur le processeur).

Les options ONNX Runtime sont réglées sans « arena » mémoire ni `mem_pattern` : avec
les valeurs par défaut, l'inférence de ce modèle monte à 5 ou 6 Go de RAM.

Un journal est écrit dans `%LOCALAPPDATA%\Mixage\mixage.log` (Windows) ou
`~/.cache/Mixage/mixage.log` : périphérique utilisé, temps par morceau, replis.
