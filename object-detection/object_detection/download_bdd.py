"""
Download BDD100K images and labels.

Called automatically by ingest_bdd.py on first run — no manual steps needed.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

_IMAGES_URL = "https://archive.org/download/bdd100k/bdd100k_images.zip"
_LABELS_URL = "https://archive.org/download/bdd100k/bdd100k_labels.zip"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest} …")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total):
        pct = min(count * block_size / total * 100, 100) if total > 0 else 0
        print(f"\r  {pct:.0f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def _find_dir_with_files(root: Path, glob: str) -> Path | None:
    """Return the first directory (including root) that contains files matching glob."""
    for p in [root, *sorted(root.rglob("*"))]:
        if p.is_dir() and any(p.glob(glob)):
            return p
    return None


def ensure_dataset(data_root: Path) -> tuple[Path, Path]:
    """
    Download and extract BDD100K images + labels into data_root if not already present.

    Both zips extract into the same subdirectory (100k/), so we locate labels first
    and then expect images co-located there. Returns (image_root, annotation_root).
    """
    zip_dir = data_root / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    # Labels
    annotation_root = _find_dir_with_files(data_root, "train/*.json")
    if annotation_root is None:
        zip_path = zip_dir / "bdd100k_labels.zip"
        if not zip_path.exists():
            _download(_LABELS_URL, zip_path)
        print(f"Extracting {zip_path.name} …")
        _extract_zip(zip_path, data_root)
        zip_path.unlink()
        annotation_root = _find_dir_with_files(data_root, "train/*.json")
        if annotation_root is None:
            raise RuntimeError(f"Could not find train/*.json anywhere under {data_root}")
    else:
        print(f"Labels already present at {annotation_root}")

    # Images — co-located with labels after a clean extraction
    if any((annotation_root / "train").glob("*.jpg")):
        image_root = annotation_root
        print(f"Images already present at {image_root}")
    else:
        image_root = _find_dir_with_files(data_root, "train/*.jpg")
        if image_root is None:
            zip_path = zip_dir / "bdd100k_images_100k.zip"
            if not zip_path.exists():
                _download(_IMAGES_URL, zip_path)
            print(f"Extracting {zip_path.name} …")
            _extract_zip(zip_path, data_root)
            zip_path.unlink()
            image_root = _find_dir_with_files(data_root, "train/*.jpg")
            if image_root is None:
                raise RuntimeError(f"Could not find train/*.jpg anywhere under {data_root}")
        print(f"Images present at {image_root}")

    try:
        zip_dir.rmdir()
    except OSError:
        pass

    return image_root, annotation_root
