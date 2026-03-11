#!/usr/bin/env python3
"""
Download and prepare the HALPE dataset with 26 body keypoints (AlphaPose HALPE-26 convention).

Assumptions:
- HALPE annotations are COCO-style JSON with a "keypoints" array per person (x,y,v triplets).
- The first 26 keypoints are body keypoints (Nose, LEye, REye, ... LHeel, RHeel) per
  https://github.com/Fang-Haoshu/Halpe-FullBody
- Train/val annotation and image URLs may require manual download from Halpe-FullBody (Baidu/Google);
  URLs are configurable at the top; set to None to skip download and use existing files.
- Script is idempotent: re-run with same output-dir skips existing files unless --force is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

# Optional: requests and tqdm for downloads (install if missing for download features)
try:
    import requests
except ImportError:
    requests = None  # type: ignore
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# -----------------------------------------------------------------------------
# Configurable URLs (edit as needed; set to None to skip download)
# Google Drive links use direct-download form (uc?export=download&id=...)
# Sources: https://github.com/Fang-Haoshu/Halpe-FullBody
# -----------------------------------------------------------------------------
HALPE_URLS: dict[str, str | None] = {
    "train_annot": "https://drive.google.com/uc?export=download&id=13vj8H0GZ9yugLPhVVWV9fcH-3RyW5Wk5",
    "val_annot": "https://drive.google.com/uc?export=download&id=1FdyCgro2t9_nOhTlMPjEf3c0aLOz9wi6",
    "train_images": "https://drive.google.com/uc?export=download&id=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
}

# Expected filenames for archives / annotations (used when saving)
HALPE_FILENAMES: dict[str, str] = {
    "train_annot": "halpe_train_v1.json",
    "val_annot": "halpe_val_v1.json",
    "train_images": "hico_20160224_det.tar.gz",
    "val_images": "val2017.zip",
}

# Number of body keypoints to keep (AlphaPose HALPE-26)
NUM_BODY_KEYPOINTS = 26
KEYPOINT_VALUES = NUM_BODY_KEYPOINTS * 3  # 78

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def create_dirs(output_dir: Path, tmp_dir: Path) -> None:
    """Create output and temp directory structure."""
    dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "annotations" / "original",
        output_dir / "annotations" / "halpe26",
        output_dir / "metadata",
        tmp_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    LOG.info("Created directory structure under %s", output_dir)


def download_file(
    url: str,
    dest: Path,
    force: bool = False,
    session: requests.Session | None = None,
) -> bool:
    """
    Download url to dest with optional resume. Returns True on success.
    Uses requests if available; otherwise falls back to urllib.
    """
    if dest.exists() and not force:
        LOG.info("Skipping (exists): %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or (requests.Session() if requests else None)
    if sess is None:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "gait-analysis-halpe26/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0)) or None
                data = resp.read()
            dest.write_bytes(data)
            LOG.info("Downloaded %s -> %s", url, dest)
            return True
        except Exception as e:
            LOG.error("Download failed: %s", e)
            return False
    try:
        r = sess.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f:
            if tqdm and total:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=total // 8192, unit="KB", desc=dest.name):
                    f.write(chunk)
            else:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        LOG.info("Downloaded %s -> %s", url, dest)
        return True
    except Exception as e:
        LOG.error("Download failed: %s", e)
        return False


def extract_zip(archive: Path, out_dir: Path, force: bool = False, flatten_one_dir: bool = False) -> bool:
    """
    Extract zip archive to out_dir. Returns True on success.
    If flatten_one_dir is True and the zip has a single top-level directory, move its contents into out_dir.
    """
    if not archive.exists():
        LOG.warning("Archive not found: %s", archive)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            for name in names:
                if name.startswith("/") or ".." in name:
                    LOG.error("Unsafe archive entry: %s", name)
                    return False
            zf.extractall(out_dir)
        if flatten_one_dir:
            subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                inner = subdirs[0]
                for p in inner.iterdir():
                    shutil.move(str(p), str(out_dir / p.name))
                inner.rmdir()
                LOG.info("Flattened single top-level dir in %s", out_dir)
        LOG.info("Extracted %s -> %s", archive, out_dir)
        return True
    except zipfile.BadZipFile as e:
        LOG.error("Invalid zip: %s", e)
        return False
    except Exception as e:
        LOG.error("Extract failed: %s", e)
        return False


def extract_tar_gz(archive: Path, out_dir: Path, flatten_one_dir: bool = True) -> bool:
    """Extract .tar.gz archive to out_dir. If flatten_one_dir, move single top-level dir contents into out_dir."""
    if not archive.exists():
        LOG.warning("Archive not found: %s", archive)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
        if flatten_one_dir:
            subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                inner = subdirs[0]
                for p in inner.iterdir():
                    shutil.move(str(p), str(out_dir / p.name))
                inner.rmdir()
                LOG.info("Flattened single top-level dir in %s", out_dir)
        LOG.info("Extracted %s -> %s", archive, out_dir)
        return True
    except Exception as e:
        LOG.error("Extract tar.gz failed: %s", e)
        return False


def validate_zip(archive: Path) -> bool:
    """Validate zip file integrity. Returns True if valid."""
    if not archive.exists():
        return False
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                LOG.warning("Zip test failed at: %s", bad)
                return False
        return True
    except Exception as e:
        LOG.warning("Zip validation error: %s", e)
        return False


def filter_annotation_to_halpe26(ann: dict[str, Any]) -> dict[str, Any] | None:
    """
    Keep only the first 26 keypoints (78 values) from a person annotation.
    Returns a new dict or None if annotation is invalid (skipped).
    """
    keypoints = ann.get("keypoints")
    if not isinstance(keypoints, list):
        LOG.debug("Skipping ann id=%s: keypoints not a list", ann.get("id"))
        return None
    if len(keypoints) < KEYPOINT_VALUES:
        LOG.debug("Skipping ann id=%s: keypoints length %d < %d", ann.get("id"), len(keypoints), KEYPOINT_VALUES)
        return None
    out = ann.copy()
    out["keypoints"] = keypoints[:KEYPOINT_VALUES]
    out["num_keypoints"] = NUM_BODY_KEYPOINTS
    return out


def process_coco_json(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    """
    Load COCO-style JSON, filter each person annotation to 26 keypoints, save.
    Returns (num_images, num_valid_annotations, num_skipped).
    """
    if not input_path.exists():
        LOG.error("Annotation file not found: %s", input_path)
        return 0, 0, 0
    with open(input_path) as f:
        data = json.load(f)
    images = data.get("images", [])
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    num_skipped = 0
    filtered_annotations = []
    for ann in annotations:
        filtered = filter_annotation_to_halpe26(ann)
        if filtered is None:
            num_skipped += 1
            continue
        if len(filtered["keypoints"]) != KEYPOINT_VALUES:
            num_skipped += 1
            continue
        filtered_annotations.append(filtered)
    out = {
        "images": images,
        "categories": categories,
        "annotations": filtered_annotations,
    }
    if "info" in data:
        out["info"] = data["info"]
    if "licenses" in data:
        out["licenses"] = data["licenses"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    LOG.info("Wrote %d images, %d annotations (skipped %d) -> %s", len(images), len(filtered_annotations), num_skipped, output_path)
    return len(images), len(filtered_annotations), num_skipped


def verify_image_references(annotations_dir: Path, images_dir: Path, json_name: str) -> bool:
    """Check that image file_names referenced in JSON exist under images_dir. Log warnings only."""
    json_path = annotations_dir / json_name
    if not json_path.exists():
        return True
    with open(json_path) as f:
        data = json.load(f)
    images = data.get("images", [])
    missing = 0
    for im in images:
        fname = im.get("file_name", "")
        if not fname:
            continue
        full = images_dir / fname
        if not full.exists():
            missing += 1
    if missing:
        LOG.warning("Image references: %d file_name(s) not found under %s", missing, images_dir)
    return True


def write_metadata(
    output_dir: Path,
    manifest_entries: list[dict[str, Any]],
) -> None:
    """Write metadata JSONs: download_manifest, keypoint_info, class_info."""
    meta = output_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "download_manifest.json", "w") as f:
        json.dump({"entries": manifest_entries}, f, indent=2)
    with open(meta / "keypoint_info.json", "w") as f:
        json.dump({
            "total_keypoints": NUM_BODY_KEYPOINTS,
            "format": ["x", "y", "visibility"],
            "source": "first 26 HALPE keypoints for AlphaPose-style body pose",
            "keypoint_names": [
                "Nose", "LEye", "REye", "LEar", "REar",
                "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
                "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
                "Head", "Neck", "Hip",
                "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
            ],
        }, f, indent=2)
    with open(meta / "class_info.json", "w") as f:
        json.dump({"classes": [{"id": 1, "name": "person"}]}, f, indent=2)
    LOG.info("Wrote metadata under %s", meta)


def run_downloads(
    output_dir: Path,
    tmp_dir: Path,
    force: bool,
    cleanup: bool,
) -> list[dict[str, Any]]:
    """Download assets per HALPE_URLS; extract to output_dir. Returns manifest entries."""
    manifest: list[dict[str, Any]] = []
    for key, url in HALPE_URLS.items():
        if not url:
            LOG.info("Skipping %s (no URL)", key)
            continue
        fname = HALPE_FILENAMES.get(key, key)
        dest = tmp_dir / fname
        if not download_file(url, dest, force=force):
            LOG.warning("Download failed for %s", key)
            continue
        manifest.append({
            "key": key,
            "url": url,
            "filename": fname,
            "downloaded_to": str(dest),
        })
        if "images" in key:
            if "train" in key:
                extract_to = output_dir / "images" / "train"
            else:
                extract_to = output_dir / "images" / "val"
            if dest.suffix.lower() == ".zip":
                if not validate_zip(dest):
                    LOG.warning("Zip validation failed for %s; extracting anyway", dest)
                flatten = "val" in key  # e.g. val2017.zip has val2017/ folder
                extract_zip(dest, extract_to, force=force, flatten_one_dir=flatten)
            elif dest.name.endswith(".tar.gz") or dest.suffix.lower() == ".gz":
                extract_tar_gz(dest, extract_to, flatten_one_dir=True)
            if cleanup and dest.exists():
                dest.unlink()
                LOG.debug("Removed temp %s", dest)
        elif "annot" in key:
            original_dir = output_dir / "annotations" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, original_dir / fname)
            if cleanup and dest.exists():
                dest.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HALPE dataset and prepare 26-keypoint (AlphaPose HALPE-26) annotations.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temp directory for downloads (default: <output-dir>/tmp)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--cleanup", action="store_true", help="Remove temp archives after extraction")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    output_dir = args.output_dir.resolve()
    tmp_dir = (args.tmp_dir or (output_dir / "tmp")).resolve()

    if output_dir.exists() and not args.force:
        pass  # idempotent: continue to create dirs and process
    create_dirs(output_dir, tmp_dir)

    manifest_entries = run_downloads(output_dir, tmp_dir, args.force, args.cleanup)

    # Process annotations: from downloads (already copied to original/) or existing files
    original_dir = output_dir / "annotations" / "original"
    total_images = total_annot = total_skipped = 0
    for split, fname in [("train", "halpe_train_v1.json"), ("val", "halpe_val_v1.json")]:
        src = original_dir / fname
        if not src.exists() and (tmp_dir / fname).exists():
            shutil.copy2(tmp_dir / fname, src)
        if src.exists():
            out_path = output_dir / "annotations" / "halpe26" / fname
            n_im, n_ann, n_skip = process_coco_json(src, out_path)
            total_images += n_im
            total_annot += n_ann
            total_skipped += n_skip
            LOG.info("Split %s: %d images, %d valid annotations, %d skipped", split, n_im, n_ann, n_skip)
            if n_ann > 0:
                with open(out_path) as f:
                    data = json.load(f)
                for a in data.get("annotations", []):
                    kp = a.get("keypoints", [])
                    if len(kp) != KEYPOINT_VALUES:
                        LOG.error("Filtered annotation has %d keypoint values (expected %d)", len(kp), KEYPOINT_VALUES)
                        return 1
            images_dir = output_dir / "images" / ("train" if "train" in fname else "val")
            verify_image_references(output_dir / "annotations" / "halpe26", images_dir, fname)

    write_metadata(output_dir, manifest_entries)

    if total_images or total_annot or total_skipped:
        print("Summary: images=%d  valid person annotations=%d  skipped=%d" % (total_images, total_annot, total_skipped))

    if args.cleanup and tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            LOG.info("Removed tmp dir %s", tmp_dir)
        except OSError as e:
            LOG.warning("Could not remove tmp dir: %s", e)

    LOG.info("Done. Output: %s", output_dir)
    print("Example command:", "python scripts/download_halpe26.py --output-dir ./datasets/halpe26 --cleanup", sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
