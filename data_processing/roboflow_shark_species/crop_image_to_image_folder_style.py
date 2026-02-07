import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re


def safe_name(s):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_")


def load_coco(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def build_image_index(images):
    return {img["id"]: img for img in images}


def build_category_map(categories):
    # map category_id -> category_name
    return {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def crop_and_save(img_path, bbox, out_path, padding=0.0):
    # bbox = [x, y, w, h]
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        iw, ih = im.size
        x, y, w, h = bbox
        # apply padding (fraction of max(w,h))
        pad = int(max(w, h) * padding)
        x1 = int(round(x)) - pad
        y1 = int(round(y)) - pad
        x2 = int(round(x + w)) + pad
        y2 = int(round(y + h)) + pad
        # clip
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(iw, x2)
        y2 = min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = im.crop((x1, y1, x2, y2))
        crop.save(out_path, quality=95)
        return True


def process_split(split_dir, annotations_file, out_base, padding=0.0, min_area=16):
    ann = load_coco(annotations_file)
    img_index = build_image_index(ann["images"])
    cat_map = build_category_map(ann["categories"])
    # iterate annotations
    for a in tqdm(ann["annotations"], desc=f"Processing {Path(annotations_file).name}"):
        img_meta = img_index.get(a["image_id"])
        if img_meta is None:
            continue
        fname = img_meta["file_name"]
        src_path = os.path.join(split_dir, fname)
        # try some common fallbacks if filename not found
        if not os.path.exists(src_path):
            # check in directory for matching basename
            bas = os.path.basename(fname)
            cand = os.path.join(split_dir, bas)
            if os.path.exists(cand):
                src_path = cand
            else:
                # try without appended rf hash (Roboflow sometimes changes names)
                alt = os.path.join(
                    split_dir, img_meta.get("extra", {}).get("name", bas)
                )
                if os.path.exists(alt):
                    src_path = alt
        if not os.path.exists(src_path):
            # skip if image missing
            tqdm.write(f"Warning: image file not found: {src_path}")
            continue
        cat_id = a["category_id"]
        cat_name = cat_map.get(cat_id, str(cat_id))
        out_dir = os.path.join(out_base, safe_name(cat_name))
        ensure_dir(out_dir)
        bbox = a["bbox"]
        area = a.get("area", bbox[2] * bbox[3])
        if area < min_area:
            continue
        out_name = f"{Path(fname).stem}__ann{a['id']}.jpg"
        out_path = os.path.join(out_dir, out_name)
        success = crop_and_save(src_path, bbox, out_path, padding=padding)
        if not success:
            tqdm.write(f"Skipping bad crop: {src_path} ann {a['id']}")


def main():
    p = argparse.ArgumentParser(
        description="Extract crops per-class from COCO-like Roboflow exports."
    )
    p.add_argument(
        "--src",
        required=True,
        help="root folder that contains train/valid/test subfolders",
    )
    p.add_argument(
        "--dst", required=True, help="output folder for ImageFolder-style crops"
    )
    p.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="padding fraction of max(w,h) to add around bbox, e.g. 0.1",
    )
    p.add_argument(
        "--min-area", type=float, default=16.0, help="minimum bbox area to keep"
    )
    args = p.parse_args()

    splits = ["train", "valid", "test"]
    for s in splits:
        split_dir = os.path.join(args.src, s)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing split directory: {split_dir}")
            continue
        # find annotation json (common Roboflow name)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            # try variants
            candidates = list(Path(split_dir).glob("*.json"))
            ann_path = None
            for c in candidates:
                with open(c, "r") as f:
                    try:
                        j = json.load(f)
                        if "annotations" in j and "images" in j:
                            ann_path = str(c)
                            break
                    except Exception:
                        continue
            if ann_path is None:
                print(f"No COCO annotations found in {split_dir}, skipping.")
                continue
        out_base = os.path.join(args.dst, s)
        ensure_dir(out_base)
        process_split(
            split_dir, ann_path, out_base, padding=args.padding, min_area=args.min_area
        )
    print("Done.")


if __name__ == "__main__":
    """
    Run with
    python data_processing/roboflow_shark_species/crop_image_to_image_folder_style.py --src /media/Linux_Data_Disk/SharkDetectionData/Roboflow_shark_species_dataset/ --dst /media/Linux_Data_Disk/SharkDetectionData/Roboflow_shark_species_dataset_cropped/
    """

    main()
