# Toy Example – Findings

## Setup

- Modell: DeiT-Small mit ICE-Klassifikationskopf (offizieller vortrainierter Checkpoint der Autoren)
- Eigenstaendige PyTorch-Reimplementierung, keine timm-Abhaengigkeit
- Eingabegroesse: 224 x 224 px, Patchgroesse 16 x 16 px → 14 x 14 = 196 Patches
- Hintergrundklasse: Index 1000 (1001. Ausgabe des MLP-Heads)
- Vordergrund: alle Patches, bei denen argmax(patch_logits) != 1000

## Ergebnisse: ILSVRC2012-Validierungsdaten (Hauptauswertung)

Drei Bilder aus dem ILSVRC2012-Validierungsdatensatz mit Bounding-Box-Ground-Truth.
Metriken auf Pixelebene, angelehnt an get_per_sample_jaccard() aus dem offiziellen ICE-Repository.
Abweichungen: 224x224 statt 512x512, Bounding Boxes statt pixelgenauer Segmentierungsmasken.

### val_00000067 – Elefant (GT: Tusker, Klasse 101)

- Vorhergesagte Klasse: 101 (Tusker) ✓
- Vordergrund-Patches: 52 / 196 (26,5 %)
- Metriken: P=0,99  R=0,42  IoU=0,42
- Vordergrund-Klassen: 101 (Tusker), 385 (Indischer Elefant), 386 (Afrikanischer Elefant)
- Beobachtung: Drei verwandte Elefanten-Synsets gleichzeitig aktiv. Modell lokalisiert
  die Objektregion korrekt, unterscheidet innerhalb der Elefantenfamilie aber nicht schaerfer.
  Hohe Precision (0,99): fast alle vorhergesagten Pixel liegen innerhalb der Bounding Box.
  Niedrigerer Recall (0,42): Bounding Box deckt ~62 % der Bildflaeche ab, Modell ist
  konservativer als die Box.

### val_00000075 – Birkhuhn (GT: Black Grouse, Klasse 83)

- Vorhergesagte Klasse: 80 (Rebhuhn/Partridge) ✗ (aber nah verwandt)
- Vordergrund-Patches: 40 / 196 (20,4 %)
- Metriken: P=0,88  R=0,62  IoU=0,57
- Vordergrund-Klassen: nur 80 (vollstaendig konsistent)
- Beobachtung: Klassifikationsfehler zwischen zwei nah verwandten Huehnervogel-Arten.
  Lokalisierung der Vogelregion gelingt trotzdem plausibel (IoU=0,57).
  Alle Patches einheitlich einer Klasse zugeordnet.

### val_00000123 – Tennisbaelle (GT: Tennis Ball, Klasse 852, 3 Instanzen)

- Vorhergesagte Klasse: 852 (Tennis Ball) ✓
- Vordergrund-Patches: 49 / 196 (25,0 %)
- Metriken: P=0,96  R=0,81  IoU=0,78
- Vordergrund-Klassen: nur 852 (vollstaendig konsistent)
- Beobachtung: Alle drei raeumlich getrennten Baelle als eine kohaerente Vordergrundregion
  erkannt. Saemtliche 49 Vordergrundpatches tragen ausschliesslich Klasse 852 – Multi-Objekt-
  Erkennung bei identischer Klasse funktioniert ohne Einschraenkungen. Bestes Ergebnis.

## Explorative Vorversuche (nicht in der Seminararbeit)

### Unsplash-Bilder (images/avocado.jpg, cherry.jpg, chili.jpg)

Drei Lebensmittelfotos vor einfarbigem Hintergrund von Unsplash (keine ImageNet-GT).
Rein visueller Abgleich, keine quantitativen Metriken moeglich.

| Bild    | Vorhergesagte Klasse | Vordergrund-Patches  |
|---------|----------------------|----------------------|
| avocado | 952                  | 20 / 196 (10,2 %)    |
| cherry  | 951                  | 19 / 196  (9,7 %)    |
| chili   | 939                  | 19 / 196  (9,7 %)    |

Ergebnis: Heatmaps konzentrieren sich auf das Objekt, Hintergrund weitgehend korrekt
als Hintergrundklasse erkannt.

### Multi-Objekt-Bilder (images/multiple_objects_in_image/)

Zwei Bilder mit mehreren Objekten verschiedener Klassen (Autos, Blumen).
Ergebnis: Modell waehlt eine dominante Klasse, alle Vordergrundpatches werden damit
eingefaerbt. Bestaetigt die in der Seminararbeit diskutierte Einschraenkung von ICE
auf Ein-Klassen-Szenarien.

## Hinweis zur Vergleichbarkeit mit dem Paper

Die hier berechneten Metriken sind nicht direkt mit den Ergebnissen von Choi et al. vergleichbar:
- Paper: 512x512 px + pixelgenaue Segmentierungsmasken (ImageNet-Segmentation-Datensatz)
- Toy Example: 224x224 px + Bounding-Box-Ground-Truth aus ILSVRC2012-XML-Annotationen
