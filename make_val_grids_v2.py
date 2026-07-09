"""
Metrik-Berechnung angelehnt an get_per_sample_jaccard() aus dem ICE-GitHub:
  - Vorhersage: argmax(patch_logits) != 1000 → binäre Vordergrundmaske
  - Hochskalierung: repeat_interleave(PATCH_SIZE) → Pixelmaske (224×224)
  - GT: Bounding Boxes aus ILSVRC-XML, auf 224×224 skaliert → binäre Pixelmaske
  - Metrik: pixel-level IoU (Jaccard) = intersection / union

Abweichungen vom originalen GitHub:
  - Bildgröße 224×224 statt 512×512 (unser Modell ist auf img_size=224 trainiert)
  - GT aus Bounding Boxes statt pixelgenauer Segmentierungsmaske
"""
import json, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import torch

BASE   = "/sessions/bold-festive-allen/mnt/Seminar/toy_example"
IMGDIR = f"{BASE}/images/imagenet validation data"
OUTDIR = f"{BASE}/results/imagenet"
import os; os.makedirs(OUTDIR, exist_ok=True)

IMG_SIZE   = 224
PATCH_SIZE = 16
GRID       = 14   # 224/16
BG         = 1000

PALETTE = [
    (230,25,75,180),(60,180,75,180),(255,225,25,180),(0,130,200,180),(245,130,48,180),
    (145,30,180,180),(70,240,240,180),(240,50,230,180),(210,245,60,180),(250,190,212,180),
    (0,128,128,180),(220,190,255,180),(170,110,40,180),(255,250,200,180),(128,0,0,180),
    (170,255,195,180),(128,128,0,180),(255,215,180,180),(0,0,128,180),(128,128,128,180),
]

# Synset → ImageNet class index (standard ILSVRC2012 mapping)
SYNSET_TO_IDX = {
    'n01871265': 101,  # tusker
    'n01795545': 80,   # black grouse
    'n04409515': 852,  # tennis ball
}

IMAGES = [
    dict(id="00000067", label="Elefant",        ext="jpeg"),
    dict(id="00000075", label="Birkhuhn",        ext="JPEG"),
    dict(id="00000123", label="Tennisbälle",     ext="JPEG"),
]

try:
    font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except:
    font_hdr = font_lbl = font_sm = ImageFont.load_default()

# ── XML parsing ───────────────────────────────────────────────────────────────
def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    orig_w, orig_h = int(size.find("width").text), int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        objects.append(dict(
            synset=obj.find("name").text,
            xmin=int(bb.find("xmin").text), ymin=int(bb.find("ymin").text),
            xmax=int(bb.find("xmax").text), ymax=int(bb.find("ymax").text),
        ))
    return orig_w, orig_h, objects

# ── GT pixel mask from bounding boxes (224×224) ───────────────────────────────
def gt_pixel_mask(orig_w, orig_h, objects):
    """Binary pixel mask at 224×224 — union of all bounding boxes."""
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for obj in objects:
        x0 = int(obj["xmin"] / orig_w * IMG_SIZE)
        y0 = int(obj["ymin"] / orig_h * IMG_SIZE)
        x1 = int(obj["xmax"] / orig_w * IMG_SIZE)
        y1 = int(obj["ymax"] / orig_h * IMG_SIZE)
        mask[y0:y1, x0:x1] = 1.0
    return mask

# ── Pixel-level IoU (mirrors get_per_sample_jaccard for single binary mask) ──
def jaccard_iou(pred_pixel, gt_pixel):
    """
    Identische Logik wie get_per_sample_jaccard() im ICE-GitHub,
    vereinfacht für eine einzelne binäre GT-Maske (kein void=255).
    pred_pixel, gt_pixel: numpy arrays (224,224), Werte 0/1
    """
    intersection = (pred_pixel * gt_pixel).sum()
    union        = ((pred_pixel + gt_pixel) > 0).sum()
    return float(intersection / union) if union > 0 else 0.0

# ── Build overlays ────────────────────────────────────────────────────────────
def build_overlays(img_path, patch_classes):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    pc = np.array(patch_classes)

    # Binary foreground mask at patch level (14×14), then upscale to 224×224
    # exactly like: attention_mask.repeat_interleave(16,dim=0).repeat_interleave(16,dim=1)
    fg_patch = (pc != BG).astype(np.float32)
    fg_pixel = fg_patch.repeat(PATCH_SIZE, axis=0).repeat(PATCH_SIZE, axis=1)  # 224×224

    # Red overlay
    arr = np.array(img, dtype=np.float32)
    red = np.zeros_like(arr); red[..., 0] = 255
    alpha = 0.45 * fg_pixel[..., None]
    ovl = Image.fromarray(np.clip(arr*(1-alpha)+red*alpha, 0, 255).astype(np.uint8))

    # Binary mask
    msk = Image.fromarray((fg_pixel * 255).astype(np.uint8)).convert("RGB")

    # Per-class colored
    fg_classes = sorted(set(pc.flatten()) - {BG})
    c2col = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(fg_classes)}
    base_rgba = img.convert("RGBA")
    col_layer = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0,0,0,0))
    d = ImageDraw.Draw(col_layer)
    for r in range(GRID):
        for c in range(GRID):
            cls = pc[r, c]
            if cls != BG:
                x0, y0 = c*PATCH_SIZE, r*PATCH_SIZE
                d.rectangle([x0, y0, x0+PATCH_SIZE-1, y0+PATCH_SIZE-1], fill=c2col[cls])
    cls_img = Image.alpha_composite(base_rgba, col_layer).convert("RGB")

    return img, ovl, msk, cls_img, fg_pixel, fg_classes, c2col

# ── Run inference + metrics ───────────────────────────────────────────────────
from models_ICE import deit_small_patch16_224
import sys; sys.path.insert(0, BASE)

model = deit_small_patch16_224(num_classes=1000, img_size=IMG_SIZE)
state = torch.load(f"{BASE}/checkpoint/0_checkpoint.pth", map_location="cpu")
model.load_state_dict(state["model"] if "model" in state else state, strict=False)
model.eval()

NORM_MEAN, NORM_STD = 0.5, 0.5

results = []
for info in IMAGES:
    img_path = f"{IMGDIR}/ILSVRC2012_val_{info['id']}.{info['ext']}"
    xml_path = f"{IMGDIR}/ILSVRC2012_val_{info['id']}.xml"

    orig_w, orig_h, objects = parse_xml(xml_path)
    gt_synsets = [o["synset"] for o in objects]
    gt_classes = [SYNSET_TO_IDX.get(s, -1) for s in set(gt_synsets)]

    # Inference
    pil = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = (np.array(pil).astype(np.float32)/255.0 - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        cls_logits, patch_logits, _ = model(tensor)
    pred_cls   = int(cls_logits.argmax(-1).item())
    patch_cls  = patch_logits.argmax(-1).reshape(GRID, GRID).numpy()

    n_fg = int((patch_cls != BG).sum())
    cnt  = Counter(patch_cls[patch_cls != BG].tolist())

    # Save metrics JSON
    met = dict(predicted_class=pred_cls,
               num_foreground_patches=n_fg,
               total_patches=GRID*GRID,
               foreground_ratio=n_fg/(GRID*GRID),
               patch_classes=patch_cls.tolist())
    with open(f"{OUTDIR}/val_{info['id']}_metrics.json","w") as f:
        json.dump(met, f, indent=2)

    # GT pixel mask + IoU
    gt_mask  = gt_pixel_mask(orig_w, orig_h, objects)
    img_out, ovl, msk, cls_img, fg_pixel, fg_classes, c2col = build_overlays(img_path, patch_cls)
    iou = jaccard_iou(fg_pixel, gt_mask)

    # Precision / Recall at pixel level
    tp = float((fg_pixel * gt_mask).sum())
    fp = float((fg_pixel * (1-gt_mask)).sum())
    fn = float(((1-fg_pixel) * gt_mask).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0

    results.append(dict(
        info=info, gt_classes=gt_classes, pred_cls=pred_cls,
        img=img_out, ovl=ovl, msk=msk, cls_img=cls_img,
        fg_classes=fg_classes, c2col=c2col, cnt=cnt,
        n_fg=n_fg, iou=iou, prec=prec, rec=rec,
        gt_mask=gt_mask,
    ))

    print(f"{info['label']}: pred={pred_cls} GT={gt_classes} | "
          f"FG={n_fg}/196 ({n_fg/196:.1%}) | {len(cnt)} Kl. | "
          f"P={prec:.2f} R={rec:.2f} IoU={iou:.2f}")

# ── Grid 1: 3×3 binary ───────────────────────────────────────────────────────
PAD, CELL, HDR_H, LBL_H = 8, IMG_SIZE, 28, 26
COL_HDRS = ["Originalbild", "Heatmap-Overlay", "Binäre Maske"]
W = 3*(CELL+PAD)+PAD
H = HDR_H + 3*(CELL+LBL_H+PAD) + PAD
c1 = Image.new("RGB", (W, H), (245,245,245))
d1 = ImageDraw.Draw(c1)
for ci, hdr in enumerate(COL_HDRS):
    d1.text((PAD+ci*(CELL+PAD)+CELL//2, HDR_H//2), hdr, fill=(50,50,50), font=font_hdr, anchor="mm")
for ri, res in enumerate(results):
    y0 = HDR_H + PAD + ri*(CELL+LBL_H+PAD)
    for ci, im in enumerate([res["img"], res["ovl"], res["msk"]]):
        c1.paste(im, (PAD+ci*(CELL+PAD), y0))
    lbl = f"{res['info']['label']}  FG {res['n_fg']}/196 · P={res['prec']:.2f} R={res['rec']:.2f} IoU={res['iou']:.2f}"
    d1.text((W//2, y0+CELL+LBL_H//2), lbl, fill=(60,60,60), font=font_sm, anchor="mm")
c1.save(f"{OUTDIR}/val_binary_grid.png")
print("Binary grid gespeichert.")

# ── Grid 2: landscape per-class ───────────────────────────────────────────────
RLW, CHH, LGH = 100, 28, 26
W2 = RLW + 3*(CELL+PAD)+PAD
H2 = CHH + 2*(CELL+LGH+PAD)+PAD
c2 = Image.new("RGB", (W2, H2), (245,245,245))
d2 = ImageDraw.Draw(c2)
for ri, lbl in enumerate(["Originalbild", "Patch-Klassen"]):
    d2.text((RLW//2, CHH+PAD+ri*(CELL+LGH+PAD)+CELL//2), lbl, fill=(50,50,50), font=font_hdr, anchor="mm")
for ci, res in enumerate(results):
    x0 = RLW + PAD + ci*(CELL+PAD)
    d2.text((x0+CELL//2, CHH//2), res["info"]["label"], fill=(50,50,50), font=font_hdr, anchor="mm")
    y0 = CHH + PAD
    c2.paste(res["img"], (x0, y0))
    d2.text((x0+CELL//2, y0+CELL+LGH//2),
            f"{res['n_fg']}/196 · {len(res['cnt'])} Kl. · IoU={res['iou']:.2f}",
            fill=(80,80,80), font=font_sm, anchor="mm")
    y1 = CHH + PAD + CELL + LGH + PAD
    c2.paste(res["cls_img"], (x0, y1))
    for i, cls in enumerate([c for c,_ in sorted(res["cnt"].items(), key=lambda x:-x[1])][:8]):
        col = res["c2col"][cls][:3]
        sx, sy = x0+i*14, y1+CELL+4
        d2.rectangle([sx, sy, sx+11, sy+11], fill=col, outline=(120,120,120))
c2.save(f"{OUTDIR}/val_class_grid.png")
print("Class grid gespeichert.")
