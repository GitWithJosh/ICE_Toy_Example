"""
Toy Example fuer ICE (Adversarial Normalization: I Can Visualize Everything)

Dieses Skript laedt ein vortrainiertes ICE-Modell (DeiT-Small-Backbone mit
patchweisem Klassifikationskopf) und wendet es auf ein einzelnes Bild an, um
die in Kapitel 5.5 der Seminararbeit beschriebene Vordergrund/Hintergrund-
Heatmap zu erzeugen.

Verwendung:
    python run_toy_example.py --image images/beispiel.jpg \
        --checkpoint checkpoint/0_checkpoint.pth \
        --output results/beispiel_heatmap.png

Checkpoint: vortrainiertes ICE-Modell der Autoren, siehe
https://drive.google.com/file/d/1zuuO40NPf-poWx-ncewj6MDV60n4LZiO/view?usp=sharing
Repository: https://github.com/Hanyang-HCC-Lab/ICE
"""

import argparse
import ast
import json
import os

import torch
import numpy as np
from PIL import Image

from models_ICE import deit_small_patch16_224

IMG_SIZE = 224
PATCH_SIZE = 16
GRID = IMG_SIZE // PATCH_SIZE  # 14
BACKGROUND_CLASS_INDEX = 1000  # 1001. Ausgabe des mlp_head (Index 1000)

# ImageNet-Normalisierung, wie im Originalrepository von ICE verwendet
NORM_MEAN = 0.5
NORM_STD = 0.5

# Standardpfade, falls keine Argumente uebergeben werden
DEFAULT_IMAGE = "images/image.png"
DEFAULT_CHECKPOINT = "checkpoint/0_checkpoint.pth"
IDX_TO_CLASS_FILE = "idx_to_class.txt"


def load_idx_to_class(path=IDX_TO_CLASS_FILE):
    """Laedt die Zuordnung von ImageNet-Index zu Klassennamen aus einer
    Datei, die ein Python-Dict-Literal enthaelt. Gibt None zurueck, wenn die
    Datei nicht vorhanden oder nicht lesbar ist."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return ast.literal_eval(f.read())
    except (ValueError, SyntaxError):
        return None


def load_image_tensor(path):
    """Laedt ein Bild, skaliert es auf 224x224 und normalisiert es,
    ohne eine Abhaengigkeit von torchvision zu benoetigen."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return img, tensor


def build_model(checkpoint_path):
    model = deit_small_patch16_224(num_classes=1000, img_size=IMG_SIZE)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # weights_only=False ist noetig, da der Checkpoint der Autoren neben
        # den Gewichten auch die Trainings-Argumente (argparse.Namespace)
        # enthaelt. Nur fuer vertrauenswuerdige Checkpoints verwenden.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint geladen: {checkpoint_path}")
        print(f"  fehlende Schluessel: {len(missing)}, unerwartete Schluessel: {len(unexpected)}")
    else:
        if checkpoint_path is not None:
            print(f"Checkpoint nicht gefunden unter: {checkpoint_path}")
        print("Kein Checkpoint geladen, Modell verwendet zufaellig initialisierte Gewichte "
              "(nur fuer einen strukturellen Test des Codes geeignet, nicht fuer eine "
              "inhaltliche Auswertung).")
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, input_tensor):
    cls_logits, patch_logits, _ = model(input_tensor)

    predicted_class = int(cls_logits.argmax(dim=-1).item())
    patch_classes = patch_logits.argmax(dim=-1).reshape(GRID, GRID)

    foreground_mask = (patch_classes != BACKGROUND_CLASS_INDEX).float()
    heatmap = foreground_mask.repeat_interleave(PATCH_SIZE, dim=0) \
                              .repeat_interleave(PATCH_SIZE, dim=1)

    num_foreground_patches = int(foreground_mask.sum().item())
    total_patches = GRID * GRID

    return {
        "predicted_class": predicted_class,
        "patch_classes": patch_classes,
        "heatmap": heatmap,
        "num_foreground_patches": num_foreground_patches,
        "total_patches": total_patches,
        "foreground_ratio": num_foreground_patches / total_patches,
    }


def save_overlay(original_img, heatmap, output_path):
    heatmap_np = heatmap.numpy()
    overlay = np.asarray(original_img).astype(np.float32).copy()

    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = 0.45 * heatmap_np[..., None]
    overlay = overlay * (1 - alpha) + red * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(overlay)
    out_img.save(output_path)

    heat_img = Image.fromarray((heatmap_np * 255).astype(np.uint8))
    heat_path = output_path.rsplit(".", 1)[0] + "_mask.png"
    heat_img.save(heat_path)
    return output_path, heat_path


def main():
    parser = argparse.ArgumentParser(description="ICE Toy Example: Einzelbild-Inferenz")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Pfad zum Eingabebild")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Pfad zum ICE-Checkpoint (0_checkpoint.pth)")
    parser.add_argument("--output", default="results/heatmap.png", help="Pfad fuer die Overlay-Ausgabe")
    parser.add_argument("--metrics-out", default=None, help="Optional: Pfad fuer eine JSON-Datei mit Kennzahlen")
    args = parser.parse_args()

    original_img, input_tensor = load_image_tensor(args.image)
    model = build_model(args.checkpoint)
    result = run_inference(model, input_tensor)

    overlay_path, mask_path = save_overlay(original_img, result["heatmap"], args.output)

    idx_to_class = load_idx_to_class()
    predicted_label = None
    if idx_to_class is not None:
        predicted_label = idx_to_class.get(result["predicted_class"])

    if predicted_label is not None:
        print(f"Vorhergesagte Klasse (ImageNet-Index): {result['predicted_class']} "
              f"({predicted_label})")
    else:
        print(f"Vorhergesagte Klasse (ImageNet-Index): {result['predicted_class']}")
    print(f"Patches im Vordergrund: {result['num_foreground_patches']} / {result['total_patches']} "
          f"({result['foreground_ratio']:.1%})")
    print(f"Overlay gespeichert unter: {overlay_path}")
    print(f"Binaere Maske gespeichert unter: {mask_path}")

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({
                "predicted_class": result["predicted_class"],
                "predicted_label": predicted_label,
                "num_foreground_patches": result["num_foreground_patches"],
                "total_patches": result["total_patches"],
                "foreground_ratio": result["foreground_ratio"],
                "patch_classes": result["patch_classes"].tolist(),
            }, f, indent=2)
        print(f"Kennzahlen gespeichert unter: {args.metrics_out}")


if __name__ == "__main__":
    main()
