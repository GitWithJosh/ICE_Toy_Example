# ICE Toy Example

Eigenstaendige, lauffaehige Umsetzung des Toy Examples fuer ICE
(Adversarial Normalization: I Can Visualize Everything, Choi et al., CVPR 2023),
beschrieben in Kapitel 5.5 der Seminararbeit.

## Inhalt

- `models_ICE.py` – eigenstaendige Neuimplementierung der ICE-Modellklasse
  (DeiT-Small-Backbone + patchweiser Klassifikationskopf, 1001 Ausgaben =
  1000 ImageNet-Klassen + 1 Hintergrundklasse), ohne timm-Abhaengigkeit
- `run_toy_example.py` – Inferenzskript fuer ein einzelnes Bild
- `make_val_grids_v2.py` – Auswertungsskript fuer die ILSVRC2012-Validierungsbilder
  (erzeugt val_binary_grid.png, val_class_grid.png und Metrik-JSONs)
- `requirements.txt` – minimale Abhaengigkeiten (torch, numpy, Pillow)
- `checkpoint/0_checkpoint.pth` – vortrainierter ICE-Checkpoint der Autoren
  (DeiT-Small, bereitgestellt ueber Google Drive; s. Seminararbeit Fussnote)

### images/

| Unterordner                | Inhalt                                              |
|----------------------------|-----------------------------------------------------|
| `imagenet_validation/`     | 3 ILSVRC2012-Validierungsbilder + XML-Annotationen  |
| `multiple_objects_in_image/` | 2 ImageNet-Bilder mit mehreren Objekten (Exploration) |
| `avocado.jpg` etc.         | Unsplash-Fotos (Exploration, keine GT verfuegbar)   |

### results/

| Unterordner          | Inhalt                                                        |
|----------------------|---------------------------------------------------------------|
| `imagenet/`          | Hauptergebnisse: val_binary_grid.png, val_class_grid.png, JSON-Metriken |
| `unsplash/`          | Ergebnisse der Unsplash-Vorversuche                          |
| `multiple_objects_in_image/` | Ergebnisse der Multi-Objekt-Exploration              |

## Hauptergebnisse (ILSVRC2012-Validierungsdaten)

| Bild          | GT-Klasse | Vorhersage       | FG-Patches   | P    | R    | IoU  |
|---------------|-----------|------------------|--------------|------|------|------|
| val_00000067  | 101 Tusker | 101 Tusker ✓    | 52/196 (27%) | 0,99 | 0,42 | 0,42 |
| val_00000075  | 83 Black Grouse | 80 Partridge ✗ | 40/196 (20%) | 0,88 | 0,62 | 0,57 |
| val_00000123  | 852 Tennis Ball | 852 Tennis Ball ✓ | 49/196 (25%) | 0,96 | 0,81 | 0,78 |

Metriken auf Pixelebene, angelehnt an `get_per_sample_jaccard()` aus dem offiziellen
ICE-Repository. Nicht direkt mit den Paper-Ergebnissen vergleichbar (224x224 statt
512x512, Bounding Boxes statt pixelgenauer Segmentierungsmasken). Details: findings.md

## Schnellstart

```bash
pip install -r requirements.txt

# Einzelbild
python run_toy_example.py \
    --image images/imagenet_validation/ILSVRC2012_val_00000067.jpeg \
    --checkpoint checkpoint/0_checkpoint.pth \
    --output results/out_heatmap.png

# Validierungs-Grids + Metriken (alle 3 Bilder)
python make_val_grids_v2.py
```

## Quellen

- ICE Paper: Choi et al., CVPR 2023 – https://doi.org/10.1109/CVPR52729.2023.01166
- Offizielles Repository: https://github.com/Hanyang-HCC-Lab/ICE (MIT-Lizenz)
- ILSVRC2012-Validierungsdaten: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
