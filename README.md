# Mixage

Application de bureau (Windows) qui retire la voix d'une chanson et garde
uniquement la musique. Tout tourne en local, sans envoyer le fichier en ligne.

Deux modèles au choix dans l'interface :

| Modèle | Qualité | Vitesse (i5 4 cœurs, processeur) | Carte graphique |
|---|---|---|---|
| **Qualité – HT-Demucs** (`htdemucs_ft_vocals`) | la meilleure | ≈ 0,8 × la durée de la chanson (3 min pour 5 min) | oui, WebGPU |
| **Rapide – Spleeter** (2 stems, Deezer) | correcte, voix un peu moins bien effacée | ≈ 1/40 de la durée (7 s pour 5 min) | inutile |

- ONNX Runtime, pas de PyTorch : l'exécutable reste léger.
- HT-Demucs : export « core » (monteslu/htdemucs-ft-webgpu, MIT), transformée de Fourier en
  numpy (`mixage/dsp.py`), modèle converti en fp32 complet par `tools/convert_fp32.py`.
- Spleeter : export ONNX du projet sherpa-onnx (k2-fsa), masque de Wiener appliqué
  en numpy (`mixage/spleeter.py`). Les fréquences au-dessus de 11 kHz ne sont pas traitées
  par ce modèle : elles restent telles quelles dans l'instrumental.
- Instrumental = mix original − voix estimée (même taux d'échantillonnage que l'entrée).
- Entrées acceptées : mp3, wav, flac, ogg… Sortie : WAV ou MP3, à côté du fichier d'origine
  (`<nom> (instrumental).wav`).

## Utilisation (développement)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m mixage                                  # interface graphique
.venv/bin/python -m mixage chanson.mp3                      # ligne de commande (HT-Demucs, WAV)
.venv/bin/python -m mixage chanson.mp3 --model spleeter --mp3
.venv/bin/python -m mixage chanson.mp3 --device both        # carte graphique + processeur
.venv/bin/python -m mixage --info                           # accélérateurs détectés
.venv/bin/python -m mixage --speed-test
```

Au premier lancement, les modèles absents sont téléchargés dans le dossier utilisateur
(`%LOCALAPPDATA%\Mixage\models` sous Windows, `~/.cache/Mixage/models` sous Linux) depuis les
Releases `model-v1` (HT-Demucs fp32, 175 Mo) et `model-v2` (Spleeter, 2 × 39 Mo) de ce dépôt,
sauf s'ils sont déjà dans `./models/` ou intégrés à l'exécutable.

## Exécutable Windows

Le workflow GitHub Actions `.github/workflows/build-windows.yml` construit `Mixage.exe`
(PyInstaller, un seul fichier, modèles intégrés) à chaque push sur `main` et publie
une Release à chaque tag `v*`. Pour construire à la main sous Windows :

```bat
pip install -r requirements.txt pyinstaller onnx
mkdir models
curl -L -o models\htdemucs_ft_vocals_safe16.onnx https://huggingface.co/monteslu/htdemucs-ft-webgpu/resolve/main/htdemucs_ft_vocals_safe16.onnx
python tools\convert_fp32.py models\htdemucs_ft_vocals_safe16.onnx models\htdemucs_ft_vocals_fp32.onnx
curl -L -o models\spleeter_vocals.onnx https://github.com/davidc4489-crypto/mixage/releases/download/model-v2/spleeter_vocals.onnx
curl -L -o models\spleeter_accompaniment.onnx https://github.com/davidc4489-crypto/mixage/releases/download/model-v2/spleeter_accompaniment.onnx
pyinstaller mixage.spec
```

Le résultat est dans `dist\Mixage.exe`. Windows SmartScreen affichera un avertissement
car l'exécutable n'est pas signé : « Informations complémentaires » puis « Exécuter quand même ».

## Carte graphique (WebGPU)

HT-Demucs peut tourner sur la carte graphique grâce au plugin **WebGPU** d'ONNX Runtime
(`onnxruntime-ep-webgpu`, Microsoft) : Direct3D 12 sous Windows (cartes NVIDIA, AMD, Intel,
intégrées ou dédiées, sans pilote supplémentaire), Vulkan sous Linux. C'est le moteur pour
lequel ce modèle a été exporté ; vérifié exact (écart 4·10⁻⁵ avec le processeur) sur un
Intel UHD 620.

L'ancien fournisseur DirectML donnait un résultat faux avec ce modèle quelle que soit la
configuration ; il n'est plus livré dans l'exécutable (le code le gère encore si le paquet
`onnxruntime-directml` est installé).

Choix dans l'interface (« Calcul ») ou en ligne de commande (`--device`) :

- **Automatique** : carte graphique si elle donne un résultat juste ; si la machine a au moins
  12 Go de mémoire, le processeur travaille en même temps sur d'autres morceaux.
- **Carte graphique + processeur ensemble** (`both`), **Carte graphique seule** (`gpu`),
  **Processeur seul** (`cpu`).

Sécurités : le premier morceau calculé sur la carte est comparé au même morceau calculé sur
le processeur ; si l'écart dépasse 5 %, la configuration suivante est essayée (WebGPU NHWC,
WebGPU sans optimisation, puis DirectML si présent) et celle qui donne un résultat juste est
mémorisée (`%LOCALAPPDATA%\Mixage\gpu_config.json`). Si aucune ne convient, ou si la carte
échoue ou dépasse 30 s sur un morceau en cours de route, le morceau est redonné au processeur
qui termine seul. `Mixage.exe --info` affiche les accélérateurs et cartes détectés ; le bouton
« Test de vitesse » chronomètre un morceau de 7,8 s sur la carte, sur le processeur et avec
Spleeter.

## Performances et mémoire

Mesuré sur un portable i5-8365U (4 cœurs, Intel UHD 620, 15 Go) :

| | 30 s d'audio | pic mémoire |
|---|---|---|
| HT-Demucs, processeur seul | 24 s | 2,6 Go |
| HT-Demucs, UHD 620 + processeur | 27 s (dont 10 s de vérification) | 3,1 Go |
| Spleeter, processeur | 1,8 s | 0,5 Go |

Une carte graphique dédiée est bien plus rapide que l'UHD 620 (qui va moins vite que le
processeur) ; la vérification du premier morceau coûte un morceau de processeur en plus.

Les options ONNX Runtime sont réglées sans « arena » mémoire ni `mem_pattern` : avec
les valeurs par défaut, l'inférence de ce modèle monte à 5 ou 6 Go de RAM.

Un journal est écrit dans `%LOCALAPPDATA%\Mixage\mixage.log` (Windows) ou
`~/.cache/Mixage/mixage.log` : périphérique utilisé, écart carte/processeur, temps par
morceau, replis.
