# ICE Toy Example

Dieser Ordner enthaelt eine lauffaehige, eigenstaendige Umsetzung des in
Kapitel 5.5 der Seminararbeit beschriebenen Toy Examples fuer ICE
(Adversarial Normalization: I Can Visualize Everything, Choi et al., 2023).

## Inhalt

- `models_ICE.py`: eigenstaendige Neuimplementierung der ICE-Modellklasse
  (DeiT-Small-Backbone mit zusaetzlichem patchweisem Klassifikationskopf,
  1001 Ausgaben = 1000 ImageNet-Klassen + 1 Hintergrundklasse). Funktional
  identisch zur Modellklasse aus dem offiziellen Repository
  (https://github.com/Hanyang-HCC-Lab/ICE, MIT-Lizenz), aber ohne
  Abhaengigkeit von der timm-Bibliothek, damit der Code mit minimalen
  Voraussetzungen lauffaehig ist. Der vortrainierte Checkpoint laedt mit
  dieser Implementierung ohne fehlende oder unerwartete Schluessel
  (siehe unten), was die strukturelle Korrektheit bestaetigt.
- `run_toy_example.py`: Inferenzskript fuer ein einzelnes Bild. Laedt
  Modell und Checkpoint, fuehrt die patchweise Klassifikation durch und
  erzeugt eine Vordergrund/Hintergrund-Heatmap nach dem in der Arbeit
  beschriebenen Verfahren (Argmax ueber 1001 Klassen pro Patch,
  Hintergrundklasse = Index 1000).
- `requirements.txt`: minimale Abhaengigkeiten (torch, numpy, Pillow).
- `checkpoint/0_checkpoint.pth`: der vortrainierte ICE-Checkpoint der
  Autoren (DeiT-Small-Backbone).
- `images/`: drei reale Testfotos von Lebensmitteln vor einfarbigem
  Hintergrund (`avocado.jpg`, `cherry.jpg`, `chili.jpg`).
- `results/`: Ausgaben der tatsaechlich durchgefuehrten Testlaeufe fuer
  alle drei Bilder (Overlay-Heatmap, binaere Maske, JSON-Kennzahlen je
  Bild).

## Tatsaechlich durchgefuehrter Testlauf

Mit dem echten Checkpoint wurde `run_toy_example.py` fuer alle drei
Testbilder erfolgreich ausgefuehrt, zum Beispiel:

```
python run_toy_example.py --image images/avocado.jpg \
    --checkpoint checkpoint/0_checkpoint.pth \
    --output results/avocado_heatmap.png \
    --metrics-out results/avocado_metrics.json
```

Ergebnis fuer alle drei Bilder:

- Der Checkpoint wurde in jedem Lauf ohne fehlende oder unerwartete
  Schluessel geladen (`fehlende Schluessel: 0, unerwartete Schluessel: 0`),
  was bestaetigt, dass die hier verwendete Modellklasse exakt der
  Architektur entspricht, mit der der Checkpoint trainiert wurde.

| Bild    | Vorhergesagte ImageNet-Klasse | Vordergrund-Patches |
|---------|-------------------------------|----------------------|
| avocado | 952                           | 20 / 196 (10,2 %)    |
| cherry  | 951                           | 19 / 196 (9,7 %)     |
| chili   | 939                           | 19 / 196 (9,7 %)     |

Die erzeugten Heatmaps (`results/<name>_heatmap.png`) zeigen in allen drei
Faellen, dass die als Vordergrund erkannten Patches ueberwiegend auf dem
jeweils abgebildeten Lebensmittel liegen, also dem einzigen klar
abgegrenzten Objekt im Bild. Vereinzelt wurde dabei auch ein bis zwei
isolierte Patches außerhalb des Objekts als Vordergrund erkannt (zum
Beispiel am Bildrand), was angesichts des einfarbigen, aber nicht
vollstaendig homogenen Hintergrunds der Fotos plausibel ist. Die binaeren
Masken (`results/<name>_heatmap_mask.png`) zeigen diese Aufteilung in
Schwarz und Weiss.

Dieses Ergebnis bestaetigt, dass die patchweise Klassifikation von ICE
tatsaechlich zwischen einem klar abgegrenzten Objekt und seinem
Hintergrund unterscheidet, auch wenn die verwendeten Fotos nicht aus dem
ImageNet-Trainingsdatensatz stammen, fuer den der Checkpoint trainiert
wurde, und auch wenn die vorhergesagten ImageNet-Klassen nicht in jedem
Fall exakt der dargestellten Lebensmittelart entsprechen.

## Eigene Bilder verwenden

Um das Beispiel mit einem weiteren eigenen Foto zu wiederholen:

1. Ein Bild mit einem klar erkennbaren Objekt nach `images/` legen.
2. Skript ausfuehren:
   ```
   python run_toy_example.py --image images/<ihr_bild>.jpg \
       --checkpoint checkpoint/0_checkpoint.pth \
       --output results/<name>_heatmap.png \
       --metrics-out results/<name>_metrics.json
   ```
3. Das Skript gibt die vorhergesagte ImageNet-Klasse sowie den Anteil der
   Patches im Vordergrund aus und speichert ein Overlay-Bild
   (`<name>_heatmap.png`) sowie eine binaere Maske
   (`<name>_heatmap_mask.png`).
