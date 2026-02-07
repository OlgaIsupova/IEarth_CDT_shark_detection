"""
Inference script: load a classifier checkpoint (best.pth / final.pth), read Sharktrack CSV,
crop detections from the corresponding videos and classify each crop. Outputs a CSV with
two new columns: species and species_confidence.

Usage example:
python /home/DanilOlyaMark/work/SharkDetection/scripts/roboflow_shark_species/classifier_inference.py \
  --ckpt ./weights/best.pth \
  --label-map ./weights/label_map.json \
  --track-csv /data/Sharktrack/outputs/input_videos_processedv5/output.csv \
  --videos-root /data/Sharktrack/input_videos \
  --out-csv /data/Sharktrack/outputs/input_videos_processedv5/output_with_species.csv \
  --device cuda \
  --reject-threshold 0.4
"""

import json
import os
from pathlib import Path
import argparse
import csv
from collections import defaultdict
import re

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
    except Exception as e:
        return None


def get_fps_ffmpeg(video_path):
    """Fallback method to get FPS using ffmpeg"""
    try:
        result = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-i", video_path],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # Look for "fps" in the output
        match = re.search(r"(\d+(?:\.\d+)?)\s*fps", result)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def read_frame_ffmpeg(video_path, frame_idx, fps=None, timeout=12):
    try:
        if fps and fps > 0:
            # Use timestamp-based seeking when fps is available
            t = (frame_idx - 1) / fps
        else:
            # Fallback: use frame-based seeking without fps (slower but more reliable)
            # Use fps estimate of 30 as default fallback
            default_fps = 30.0
            t = (frame_idx - 1) / default_fps

        # Try with different approaches
        approaches = [
            # Approach 1: Standard PNG output
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            # Approach 2: PPM output (faster, simpler format)
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "pipe:1",
            ],
            # Approach 3: JPEG output (even more compatible)
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
        ]

        for cmd in approaches:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                img_bytes, err_bytes = proc.communicate(timeout=timeout)
                if img_bytes and len(img_bytes) > 0:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return img
            except Exception:
                continue

        return None
    except Exception as e:
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
    p.add_argument("--track-csv", required=True, help="sharktrack output csv")
    p.add_argument(
        "--videos-root",
        required=True,
        help="root folder that contains input videos (matching video_name column)",
    )
    p.add_argument(
        "--out-csv", required=True, help="output csv path with species columns added"
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
        help="optional folder to save annotated frames with bboxes and species labels",
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

    rows, fieldnames = read_track_csv(args.track_csv)
    grouped = group_rows_by_video_and_frame(
        rows, video_col="video_name", frame_col="frame"
    )

    out_rows = []
    # add new columns
    out_fieldnames = list(fieldnames) + ["species", "species_confidence"]

    vis_root = Path(args.vis_dir) if args.vis_dir else None

    for video_name, frames in tqdm(grouped.items(), desc="Videos"):
        video_path = None
        # try full path first (video_name may contain extension only)
        cand = os.path.join(args.videos_root, video_name)
        if os.path.exists(cand):
            video_path = cand
        else:
            # try finding file by name anywhere under videos_root
            for ext in ("", ".MP4", ".mp4", ".MOV", ".mov", ".avi"):
                cand2 = os.path.join(args.videos_root, video_name + ext)
                if os.path.exists(cand2):
                    video_path = cand2
                    break
        if not video_path:
            # try to walk root for matching basename
            for p in Path(args.videos_root).rglob(video_name):
                video_path = str(p)
                break
        if not video_path:
            tqdm.write(
                f"Warning: video file for '{video_name}' not found under {args.videos_root}, skipping its rows."
            )
            # still append rows with empty labels
            for frame_rows in frames.values():
                for r in frame_rows:
                    r_out = dict(r)
                    r_out["species"] = ""
                    r_out["species_confidence"] = ""
                    out_rows.append(r_out)
            continue

        # prepare per-video vis dir if requested
        if vis_root:
            vis_video_dir = vis_root / Path(video_name).stem
            vis_video_dir.mkdir(parents=True, exist_ok=True)
        else:
            vis_video_dir = None

        cap = cv2.VideoCapture(video_path)
        use_ffmpeg = False
        fps = None

        if not cap.isOpened():
            tqdm.write(
                f"Warning: could not open video {video_path} with OpenCV. Falling back to ffmpeg."
            )
            use_ffmpeg = True
            # Try to get fps with ffprobe first
            fps = get_fps_ffprobe(video_path)
            if fps is None:
                # Fallback: try ffmpeg method
                fps = get_fps_ffmpeg(video_path)
            if fps is None:
                # Last resort: use default estimate
                tqdm.write(
                    f"Warning: could not determine FPS for {video_path}, using default 30 fps"
                )
                fps = 30.0
            tqdm.write(f"Video fps: {fps}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                tqdm.write(f"Warning: OpenCV returned invalid fps, trying ffprobe")
                fps = get_fps_ffprobe(video_path)
                if fps is None:
                    fps = get_fps_ffmpeg(video_path)
                if fps is None:
                    fps = 30.0
                tqdm.write(f"FPS determined: {fps}")

        # process frames in sorted order to minimize seeks
        for frame_idx in sorted(frames.keys()):
            # set to frame (CSV frame likely 1-based; cv2 uses 0-based)
            if use_ffmpeg:
                pil_img = read_frame_ffmpeg(video_path, frame_idx, fps=fps)
                if pil_img is None:
                    tqdm.write(
                        f"Warning: ffmpeg failed to extract frame {frame_idx} from {video_name} (fps={fps}, path={video_path})"
                    )
                    for r in frames[frame_idx]:
                        r_out = dict(r)
                        r_out["species"] = ""
                        r_out["species_confidence"] = ""
                        out_rows.append(r_out)
                    continue
                pil_img_full = pil_img
                w, h = pil_img_full.size
            else:
                target = max(0, frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frm = cap.read()
                if not ok or frm is None:
                    tqdm.write(
                        f"Warning: could not read frame {frame_idx} from {video_name} via OpenCV"
                    )
                    for r in frames[frame_idx]:
                        r_out = dict(r)
                        r_out["species"] = ""
                        r_out["species_confidence"] = ""
                        out_rows.append(r_out)
                    continue
                frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                h, w, _ = frm_rgb.shape
                pil_img_full = Image.fromarray(frm_rgb)

            # prepare annotation canvas (copy) and drawing tools
            if vis_video_dir:
                annot_img = pil_img_full.copy()
                draw = ImageDraw.Draw(annot_img)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
            else:
                annot_img = None
                draw = None
                font = None

            # process each detection in this frame
            for r in frames[frame_idx]:
                # expect bbox fields xmin,ymin,xmax,ymax (float)
                try:
                    xmin = float(r.get("xmin", r.get("x1", 0)))
                    ymin = float(r.get("ymin", r.get("y1", 0)))
                    xmax = float(r.get("xmax", r.get("x2", 0)))
                    ymax = float(r.get("ymax", r.get("y2", 0)))
                except Exception:
                    xmin = ymin = xmax = ymax = 0.0
                bb = clamp_bbox(xmin, ymin, xmax, ymax, w, h)
                if bb is None:
                    r_out = dict(r)
                    r_out["species"] = ""
                    r_out["species_confidence"] = ""
                    out_rows.append(r_out)
                    continue
                crop = pil_img_full.crop(bb)
                label, conf = classify_crop(
                    model, device, crop, transform, class_names, reject_threshold
                )
                r_out = dict(r)
                r_out["species"] = label
                r_out["species_confidence"] = f"{conf:.4f}"
                out_rows.append(r_out)

                # draw bbox + label on annot_img if requested
                if annot_img is not None and draw is not None:
                    x1, y1, x2, y2 = bb
                    color = (0, 200, 0) if label != "background" else (200, 0, 0)
                    # rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    # text
                    txt = f"{label} {conf:.2f}"
                    tw, th = get_text_size(draw, txt, font)
                    # position text above bbox if possible, else below
                    tx1 = x1
                    ty1 = y1 - th - 3
                    ty2 = y1
                    if ty1 < 0:
                        ty1 = y2 + 3
                        ty2 = ty1 + th + 3
                    tx2 = tx1 + tw + 6
                    # background rect
                    draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
                    text_x = tx1 + 3
                    text_y = ty1 + 1
                    draw.text((text_x, text_y), txt, fill=(255, 255, 255), font=font)

            # save annotated frame if requested
            if vis_video_dir:
                vis_path = vis_video_dir / f"{frame_idx:06d}.jpg"
                try:
                    annot_img.save(str(vis_path), quality=90)
                except Exception as e:
                    tqdm.write(
                        f"Warning: could not save annotated frame {vis_path}: {e}"
                    )

        if not use_ffmpeg:
            cap.release()

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

    # python scripts/roboflow_shark_species/classifier_inference.py --ckpt /data/Roboflow_shark_species_dataset_cropped/output/resnet50/best.pth --label-map /data/Roboflow_shark_species_dataset_cropped/output/resnet50/label_map.json --track-csv /data/Sharktrack/outputs/input_videos_processed_track_manysharks/internal_results/output.csv --videos-root /data/Sharktrack/input_videos --out-csv /data/Sharktrack/outputs/input_videos_processed_track_manysharks/internal_results/output_classified.csv --reject-threshold 0.4 --vis-dir /data/Sharktrack/outputs/input_videos_processed_track_manysharks/internal_results/Many_sharks_classification_visualisation
