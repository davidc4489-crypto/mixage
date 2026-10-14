# Mixage

Application de bureau (Windows) qui retire la voix d'une chanson et garde
uniquement la musique. Tout tourne en local, sans envoyer le fichier en ligne.

- Modèle : HT-Demucs (spécialiste voix, `htdemucs_ft_vocals`) au format ONNX,
  exécuté par ONNX Runtime. Pas de PyTorch : l'exécutable reste léger.
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

Au premier lancement, le modèle (165 Mo) est téléchargé dans le dossier
utilisateur (`%LOCALAPPDATA%\Mixage\models` sous Windows, `~/.cache/Mixage/models` sous Linux),
sauf s'il est déjà fourni dans `./models/` ou intégré à l'exécutable.

## Exécutable Windows

Le workflow GitHub Actions `.github/workflows/build-windows.yml` construit `Mixage.exe`
(PyInstaller, un seul fichier, modèle intégré) à chaque push sur `main` et publie
une Release à chaque tag `v*`. Pour construire à la main sous Windows :

```bat
pip install -r requirements.txt pyinstaller
mkdir models
curl -L -o models\htdemucs_ft_vocals_fp16weights.onnx https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx
pyinstaller mixage.spec
```

Le résultat est dans `dist\Mixage.exe`. Windows SmartScreen affichera un avertissement
car l'exécutable n'est pas signé : « Informations complémentaires » puis « Exécuter quand même ».

## Performances et mémoire

Mesuré sur un PC 8 cœurs sans GPU : environ 1,3 × la durée de la chanson
(une chanson de 3 min prend 4 min), pic mémoire 1,4 Go.

Les options ONNX Runtime sont volontairement réglées en « basse mémoire »
(`ORT_DISABLE_ALL`, pas d'arena) : avec les optimisations par défaut, l'initialisation
du modèle dépasse 7 Go de RAM et peut bloquer la machine.
