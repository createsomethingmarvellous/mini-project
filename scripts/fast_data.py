"""
Fast data: work on Colab's local disk, keep one archive per dataset on Drive.

Google Drive is very slow at reading or writing thousands of small audio
files, but fast with a few big files. So:
  - each dataset is stored on Drive as ONE archive in mini_project/archives/
  - every session unpacks the archives to Colab's local disk (/content/fast)
  - all scripts then read and write there automatically (see config.py)
  - after building new data, pack it back to Drive

Usage (in Colab):
    !python scripts/fast_data.py unpack                 # every archive on Drive
    !python scripts/fast_data.py pack eval_noisy        # after T6
    !python scripts/fast_data.py status
Archive names: see ARCHIVES below.
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

LOCAL = config.LOCAL_DATA

# archive name -> folders (glob patterns, relative to the local data folder)
ARCHIVES = {
    "asvspoof": ["asvspoof2019"],
    "noise": ["MS-SNSD/noise_train", "MS-SNSD/noise_test", "RIRS_NOISES/simulated_rirs"],
    "eval_noisy": ["generated/eval/noisy_snr*"],
    **{f"eval_{enh}": [f"generated/eval/{enh}_snr*"] for enh in config.ENHANCEMENT_MODELS},
    "train_aug": ["generated/train_aug"],
}


def archive_path(name: str) -> Path:
    # 1. Direct match in ARCHIVE_DIR
    p = config.ARCHIVE_DIR / f"{name}.tar"
    if p.exists():
        return p
    # 2. Direct match in BASE / archives
    p = config.BASE / "archives" / f"{name}.tar"
    if p.exists():
        return p
    # 3. Check recursively in all /kaggle/input datasets
    if Path("/kaggle/input").exists():
        matches = list(Path("/kaggle/input").rglob(f"{name}.tar"))
        if matches:
            return matches[0]
        # Also check for .tar.gz
        gz_matches = list(Path("/kaggle/input").rglob(f"{name}.tar.gz"))
        if gz_matches:
            return gz_matches[0]
        # Partial match
        clean_name = name.replace("_", "").lower()
        for cand in Path("/kaggle/input").rglob("*.tar"):
            if cand.stem.replace("_", "").replace("-", "").lower() == clean_name:
                return cand
    return config.ARCHIVE_DIR / f"{name}.tar"


def info_path(name: str) -> Path:
    arch = archive_path(name)
    return arch.with_suffix(".json")


def marker_path(name: str) -> Path:
    return LOCAL / f".unpacked_{name}.json"


def pack(name: str) -> None:
    folders = [p for pattern in ARCHIVES[name] for p in sorted(LOCAL.glob(pattern))]
    if not folders:
        sys.exit(f"ERROR: nothing to pack for '{name}' under {LOCAL} "
                 f"(looked for {ARCHIVES[name]}).")
    files = [f for folder in folders for f in sorted(folder.rglob("*")) if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"Packing {name}: {len(files)} files, {total / 1e9:.2f} GB -> {archive_path(name)}")

    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = archive_path(name).with_suffix(".tar.part")
    with tarfile.open(tmp, "w|") as tar, \
            tqdm(total=total, unit="B", unit_scale=True, desc=f"pack {name}",
                 mininterval=5) as bar:
        for f in files:
            tar.add(f, arcname=f.relative_to(LOCAL).as_posix(), recursive=False)
            bar.update(f.stat().st_size)
    os.replace(tmp, archive_path(name))

    info = {"files": len(files), "bytes": total,
            "folders": [p.relative_to(LOCAL).as_posix() for p in folders]}
    info_path(name).write_text(json.dumps(info, indent=1))
    marker_path(name).write_text(json.dumps(_stamp(name)))
    print(f"Done. Run the notebook's last cell (save to Drive) before closing the tab.")


def _stamp(name: str) -> dict:
    st = archive_path(name).stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def _extract(tar, member):
    try:
        tar.extract(member, LOCAL, filter="data")   # safe extraction (Python 3.12+)
    except TypeError:
        tar.extract(member, LOCAL)                  # older Python


def is_already_available(name: str) -> bool:
    """Check if the dataset is already extracted in /kaggle/input (fast, no file scan)."""
    if not Path("/kaggle/input").exists():
        return False
    kag = Path("/kaggle/input")
    if name == "asvspoof":
        # tarfiles layout: <slug>/asvspoof-*/asvspoof2019/LA/LA
        for pat in (
            "*/asvspoof*/LA/LA",
            "*/*/asvspoof*/LA/LA",    # ← tarfiles (1 extra nesting level)
            "*/asvspoof2019/LA/LA",
            "*/*/asvspoof2019/LA/LA", # ← tarfiles exact
        ):
            for p in kag.glob(pat):
                if p.is_dir() and (
                    (p / "ASVspoof2019_LA_cm_protocols").is_dir()
                    or any(c.name.startswith("ASVspoof2019_LA") for c in p.iterdir() if c.is_dir())
                ):
                    return True
    elif name == "train_aug":
        for pat in ("*/train_aug", "*/generated/train_aug"):
            for p in kag.glob(pat):
                if p.is_dir():
                    return True
    elif name == "noise":
        for pat in ("*/noise*/noise_train", "*/noise_train", "*/MS-SNSD/noise_train"):
            for p in kag.glob(pat):
                if p.is_dir():
                    return True
    elif name.startswith("eval_"):
        cond_prefix = name.replace("eval_", "")
        for pat in (f"*/{cond_prefix}_snr*", f"*/{name}"):
            for p in kag.glob(pat):
                if p.is_dir():
                    return True
    return False


def unpack(name: str, delete_archive: bool = False, force: bool = False) -> None:
    if is_already_available(name):
        print(f"  {name:<16} detected pre-extracted in /kaggle/input (0 MB local disk used!)")
        return
    arch = archive_path(name)
    if not arch.exists():
        print(f"  {name:<16} no archive on Drive yet - skipped")
        return
    marker = marker_path(name)
    if marker.exists() and json.loads(marker.read_text()) == _stamp(name):
        print(f"  {name:<16} already on the local disk")
        return

    free = shutil.disk_usage(LOCAL.parent if not LOCAL.exists() else LOCAL).free
    if arch.stat().st_size > free * 0.9 and not force:
        sys.exit(f"ERROR: not enough local disk for {name} "
                 f"({arch.stat().st_size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free). "
                 f"Tip: delete unused evaluation archives or use --delete-archive / --force.")

    count = None
    if info_path(name).exists():
        count = json.loads(info_path(name).read_text()).get("files")
    LOCAL.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(name)
    with tarfile.open(arch, "r|") as tar:
        for member in tqdm(tar, total=count, unit="file", desc=f"unpack {name}",
                           mininterval=5):
            _extract(tar, member)
    marker.write_text(json.dumps(stamp))
    if delete_archive:
        try:
            arch.unlink()
            print(f"  {name:<16} archive removed to free up disk ({stamp['size'] / 1e9:.2f} GB freed)")
        except Exception as e:
            print(f"  Could not remove archive {arch}: {e}")


def status() -> None:
    print(f"Local data folder: {LOCAL}")
    for name in ARCHIVES:
        arch = archive_path(name)
        on_drive = f"{arch.stat().st_size / 1e9:6.2f} GB on Drive/Input" if arch.exists() else "   not on Drive"
        if is_already_available(name):
            local = "/kaggle/input (ready)"
        elif marker_path(name).exists():
            local = "unpacked"
        else:
            local = "-"
        print(f"  {name:<16} {on_drive}   local: {local}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pack", "unpack", "status"])
    parser.add_argument("names", nargs="*", help=f"archives: {list(ARCHIVES)}")
    parser.add_argument("--delete-archive", action="store_true",
                        help="delete archive file right after unpacking to save disk (e.g. in Kaggle)")
    parser.add_argument("--force", action="store_true",
                        help="force unpacking even if free disk calculation is tight")
    args = parser.parse_args()
    bad = [n for n in args.names if n not in ARCHIVES]
    if bad:
        sys.exit(f"Unknown archive(s) {bad}. Choose from {list(ARCHIVES)}")

    if args.action == "status":
        status()
    elif args.action == "pack":
        if not args.names:
            sys.exit("Say what to pack, e.g.: pack eval_noisy")
        for name in args.names:
            pack(name)
    else:
        print("Unpacking archives from Drive to the fast local disk:")
        for name in args.names or ARCHIVES:
            unpack(name, delete_archive=args.delete_archive, force=args.force)
        status()


if __name__ == "__main__":
    main()
