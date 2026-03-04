#!/usr/bin/env python3
"""
Download CMU Human Foot Keypoint annotations from alternative sources.

The official CMU URLs (posefs1.perception.cs.cmu.edu) are often unavailable.
This script fetches the annotation JSONs from a GitHub mirror that hosts them.

Source: https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/
Mirror (annotations only): https://github.com/Eva20150932/coco-foot-and-leg

You still need COCO 2017 images separately (see README or --help).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import urllib.request
except Exception:
    urllib = None  # type: ignore

# GitHub mirror: repo contains person_keypoints_train2017_foot_v1.json and person_keypoints_val2017_foot_v1.json
FOOT_ANNOT_URLS = {
    "train": "https://raw.githubusercontent.com/Eva20150932/coco-foot-and-leg/main/person_keypoints_train2017_foot_v1.json",
    "val": "https://raw.githubusercontent.com/Eva20150932/coco-foot-and-leg/main/person_keypoints_val2017_foot_v1.json",
}

# Official CMU URLs (may be down)
CMU_OFFICIAL_URLS = {
    "train": "http://posefs1.perception.cs.cmu.edu/OpenPose/datasets/foot/person_keypoints_train2017_foot_v1.zip",
    "val": "http://posefs1.perception.cs.cmu.edu/OpenPose/datasets/foot/person_keypoints_val2017_foot_v1.zip",
}

COCO_IMAGE_URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
}


def download_file(url: str, dest: Path, desc: str = "file") -> bool:
    """Download url to dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; gait-analysis/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"Failed to download {desc}: {e}", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"Saved {desc} -> {dest}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download CMU Foot Keypoint annotations from alternative sources (GitHub mirror)."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/foot_cmu/annotations"),
        help="Directory to save annotation JSON files",
    )
    ap.add_argument(
        "--source",
        choices=["github", "cmu"],
        default="github",
        help="github = Eva20150932/coco-foot-and-leg mirror (JSON); cmu = official CMU (ZIP, often down)",
    )
    ap.add_argument(
        "--train-only",
        action="store_true",
        help="Download only training annotations",
    )
    ap.add_argument(
        "--val-only",
        action="store_true",
        help="Download only validation annotations",
    )
    ap.add_argument(
        "--print-urls",
        action="store_true",
        help="Print all alternative URLs and exit (for manual download)",
    )
    args = ap.parse_args()

    if args.print_urls:
        print("Alternative download routes for CMU Human Foot Keypoint annotations\n")
        print("1) Annotation JSONs (GitHub mirror, recommended):")
        print("   Train:", FOOT_ANNOT_URLS["train"])
        print("   Val:  ", FOOT_ANNOT_URLS["val"])
        print("\n2) Official CMU (ZIP, often unavailable):")
        print("   Train:", CMU_OFFICIAL_URLS["train"])
        print("   Val:  ", CMU_OFFICIAL_URLS["val"])
        print("\n3) COCO 2017 images (required; use one of these):")
        print("   Train (18GB):", COCO_IMAGE_URLS["train"])
        print("   Val (1GB):  ", COCO_IMAGE_URLS["val"])
        print("\n   Or via Academic Torrents: COCO 2017 train/val zips")
        print("   https://academictorrents.com/details/74dec1dd21ae4994dfd9069f9cb0443eb960c962")
        return

    if args.source == "cmu":
        print("CMU official URLs often fail. Use --source github or --print-urls for mirror.", file=sys.stderr)
        urls = CMU_OFFICIAL_URLS
        ext = ".zip"
    else:
        urls = FOOT_ANNOT_URLS
        ext = ".json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    if not args.val_only:
        dest = args.output_dir / f"person_keypoints_train2017_foot_v1{ext}"
        if ext == ".zip" and dest.suffix != ".zip":
            dest = args.output_dir / "person_keypoints_train2017_foot_v1.zip"
        ok = download_file(urls["train"], dest, "train annotations") and ok
    if not args.train_only:
        dest = args.output_dir / f"person_keypoints_val2017_foot_v1{ext}"
        if ext == ".zip" and dest.suffix != ".zip":
            dest = args.output_dir / "person_keypoints_val2017_foot_v1.zip"
        ok = download_file(urls["val"], dest, "val annotations") and ok

    if ok:
        print("\nNext: download COCO 2017 images (train2017.zip, val2017.zip) and run prepare_dataset with --coco-annot pointing to the train JSON.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
