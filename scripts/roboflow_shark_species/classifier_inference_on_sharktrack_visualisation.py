"""
Inference script: load a classifier checkpoint (best.pth / final.pth), read JPG images
from a folder, and classify each image. Outputs a CSV with species and species_confidence columns.

Usage example:
python scripts/roboflow_shark_species/classifier_inference_on_sharktrack_visualisation.py \
  --ckpt ./weights/best.pth \
  --label-map ./weights/label_map.json \
  --img-dir /data/Sharktrack/outputs/input_video_processedv5/internal_results/Many_sharks_first2min \
  --out-csv ./results/species_classifications.csv \
  --device cuda \
  --reject-threshold 0.4 \
  --vis-dir ./results/annotated_images
"""

import json
import os
from pathlib import Path
import argparse
import csv
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from tqdm import tqdm
import subprocess
import io
from PIL import Image, ImageDraw, ImageFont


def get_model(arch, num_classes, pretrained=False):
    if arch == "resnet50":
        m = models.resnet50(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(pretrained=pretrained)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=pretrained)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m


def load_label_map(label_map_path):
    if not label_map_path:
        return None
    with open(label_map_path, "r") as f:
        return json.load(f)


def build_transform(img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return tf


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    x1 = max(0, min(w - 1, int(round(xmin))))
    y1 = max(0, min(h - 1, int(round(ymin))))
    x2 = max(0, min(w, int(round(xmax))))
    y2 = max(0, min(h, int(round(ymax))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def classify_crop(model, device, pil_img, transform, class_names, reject_threshold=0.0):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    top_p = float(probs[top_idx])
    if reject_threshold > 0.0 and top_p < reject_threshold:
        return "background", top_p
    # if there is a background class in class_names, it will return that label normally
    return class_names[top_idx], top_p


def read_track_csv(path):
    rows = []
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            rows.append(r)
    return rows, reader.fieldnames


def group_rows_by_video_and_frame(rows, video_col="video_name", frame_col="frame"):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        video = r.get(video_col) or r.get("video_path")
        frame = int(float(r.get(frame_col, 0)))
        grouped[video][frame].append(r)
    return grouped


def get_fps_ffprobe(video_path):
    try:
        out = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if not out:
            return None
        if "/" in out:
            n, d = out.split("/")
            return float(n) / float(d)
        return float(out)
    except Exception:
        return None


def read_frame_ffmpeg(video_path, frame_idx, fps=None, timeout=12):
    try:
        if fps and fps > 0:
            t = (frame_idx - 1) / fps
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ]
        else:
            sel = f"select=eq(n\\,{frame_idx-1})"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vf",
                sel,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        img_bytes, _ = proc.communicate(timeout=timeout)
        if not img_bytes:
            return None
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def get_text_size(draw, text, font=None):
    """
    Return (width, height) for rendered text in a cross-Pillow-compatible way.
    Tries draw.textbbox -> draw.textsize -> font.getsize -> fallback estimate.
    """
    try:
        # Pillow >= 8.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass
    try:
        # older Pillow
        return draw.textsize(text, font=font)
    except Exception:
        pass
    if font is not None:
        try:
            return font.getsize(text)
        except Exception:
            pass
    # fallback rough estimate
    return (max(10, len(text) * 6), 12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", required=True, help="checkpoint .pth (best.pth / final.pth)"
    )
    p.add_argument(
        "--label-map",
        required=False,
        help="label_map.json produced at training (optional)",
    )
    p.add_argument(
        "--img-dir",
        required=True,
        help="folder containing .jpg images to classify",
    )
    p.add_argument(
        "--out-csv", required=True, help="output csv path with species columns"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--reject-threshold",
        type=float,
        default=0.0,
        help="override label_map reject_threshold if provided",
    )
    p.add_argument(
        "--vis-dir",
        default=None,
        help="optional folder to save annotated images with species labels",
    )
    args = p.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ck = torch.load(args.ckpt, map_location=device)
    # load class names from checkpoint or label-map
    class_names = ck.get("class_names")
    label_map = None
    if args.label_map and os.path.exists(args.label_map):
        label_map = load_label_map(args.label_map)
        if label_map and "classes" in label_map:
            class_names = label_map["classes"]
    if not class_names:
        raise SystemExit("No class names found in checkpoint or label_map.json")

    arch = ck.get("arch", "resnet50")
    model = get_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ck["model_state"])
    model = model.to(device)
    model.eval()

    # determine reject threshold (priority: CLI arg > label_map)
    reject_threshold = args.reject_threshold
    if reject_threshold == 0.0 and label_map:
        reject_threshold = float(label_map.get("reject_threshold", 0.0))

    transform = build_transform(args.img_size)

    # Get all .jpg files from input directory
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {args.img_dir}")

    jpg_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
    if not jpg_files:
        raise SystemExit(f"No .jpg files found in {args.img_dir}")

    print(f"Found {len(jpg_files)} images to classify")

    out_rows = []
    out_fieldnames = ["image_name", "species", "species_confidence"]

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_path in tqdm(jpg_files, desc="Processing images"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
        except Exception as e:
            tqdm.write(f"Warning: could not open image {img_path}: {e}")
            out_rows.append(
                {"image_name": img_path.name, "species": "", "species_confidence": ""}
            )
            continue

        # Classify the entire image (no bboxes)
        label, conf = classify_crop(
            model, device, pil_img, transform, class_names, reject_threshold
        )

        out_rows.append(
            {
                "image_name": img_path.name,
                "species": label,
                "species_confidence": f"{conf:.4f}",
            }
        )

        # Save annotated image if requested
        if vis_dir:
            annot_img = pil_img.copy()
            draw = ImageDraw.Draw(annot_img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            # Draw species label at the top
            txt = f"{label} {conf:.2f}"
            tw, th = get_text_size(draw, txt, font)
            tx1, ty1 = 10, 10
            tx2, ty2 = tx1 + tw + 6, ty1 + th + 6
            color = (0, 200, 0) if label != "background" else (200, 0, 0)
            draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
            draw.text((tx1 + 3, ty1 + 1), txt, fill=(255, 255, 255), font=font)

            vis_path = vis_dir / img_path.name
            try:
                annot_img.save(str(vis_path), quality=90)
            except Exception as e:
                tqdm.write(f"Warning: could not save annotated image {vis_path}: {e}")

    # write output csv
    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print("Done. Output saved to", args.out_csv)


if __name__ == "__main__":
    main()

    # python scripts/roboflow_shark_species/classifier_inference.py --ckpt /data/Roboflow_shark_species_dataset_cropped/output/resnet50/best.pth --label-map /data/Roboflow_shark_species_dataset_cropped/output/resnet50/label_map.json --track-csv /data/Sharktrack/outputs/input_videos_processedv5/internal_results/output.csv --videos-root /data/Sharktrack/input_videos --out-csv /data/Sharktrack/outputs/input_videos_processedv5/internal_results/output_classified.csv --reject-threshold 0.4
