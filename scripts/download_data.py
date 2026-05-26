"""
Download the 4 FlyWire data files from Zenodo into data/real/
Total size: ~1.1 GB  (skips the 9.5 GB synapses file)

Run from the project root:
    python3 scripts/download_data.py
"""

import urllib.request
import os
from pathlib import Path

DEST = Path(__file__).parent.parent / "data" / "real"
DEST.mkdir(parents=True, exist_ok=True)

FILES = [
    ("proofread_connections_783.feather",        "852 MB",
     "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"),
    ("per_neuron_neuropil_count_pre_783.feather", " 17 MB",
     "https://zenodo.org/records/10676866/files/per_neuron_neuropil_count_pre_783.feather?download=1"),
    ("per_neuron_neuropil_count_post_783.feather","234 MB",
     "https://zenodo.org/records/10676866/files/per_neuron_neuropil_count_post_783.feather?download=1"),
    ("proofread_root_ids_783.npy",               "  1 MB",
     "https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1"),
]


def progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size / total_size * 100, 100)
        mb_done = count * block_size / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb_done:6.0f} / {mb_total:.0f} MB", end="", flush=True)


for fname, size, url in FILES:
    dest_path = DEST / fname
    if dest_path.exists():
        print(f"[skip] {fname}  (already downloaded)")
        continue
    print(f"\n[download] {fname}  ({size})")
    print(f"  → {dest_path}")
    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print(f"\n  Done: {dest_path.stat().st_size / 1_048_576:.0f} MB written")

print("\nAll files present in", DEST)
