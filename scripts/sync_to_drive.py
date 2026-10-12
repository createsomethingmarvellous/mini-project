"""
Helper script to verify, zip, and synchronize model checkpoints
from Kaggle or local environments directly into Google Drive.

Usage:
    # 1. Zip checkpoints for direct Kaggle Output download:
    python scripts/sync_to_drive.py zip --tag B --seed 123

    # 2. Upload checkpoints directly into a Google Drive folder:
    python scripts/sync_to_drive.py upload --tag B --seed 123 --folder-id <GDRIVE_FOLDER_ID>
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
import config


import re


def clean_gdrive_id(url_or_id: str) -> str:
    """Extract clean Google Drive ID from any URL or string."""
    if not url_or_id:
        return ""
    s = str(url_or_id).strip()
    m = re.search(r'folders/([a-zA-Z0-9_-]+)', s)
    if m:
        return m.group(1)
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', s)
    if m:
        return m.group(1)
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', s)
    if m:
        return m.group(1)
    return s


def get_checkpoint_files(tag: str, seed: int):
    """Find all valid checkpoint files for a model tag and seed."""
    candidates = [
        config.CHECKPOINT_DIR / f"{tag}_seed{seed}",
        Path("/kaggle/working/mini_project/checkpoints") / f"{tag}_seed{seed}",
        Path("/kaggle/working/checkpoints") / f"{tag}_seed{seed}",
        Path("checkpoints") / f"{tag}_seed{seed}",
        Path("data/checkpoints") / f"{tag}_seed{seed}",
    ]
    ckpt_dir = None
    for cand in candidates:
        if cand.exists():
            ckpt_dir = cand
            break

    if ckpt_dir is None or not ckpt_dir.exists():
        print(f"[!] Checkpoint directory not found among candidates: {[str(c) for c in candidates]}")
        return None, []

    files = []
    for ext in ["*.pth", "*.pt", "*.txt", "*.json"]:
        files.extend(list(ckpt_dir.glob(ext)))

    return ckpt_dir, sorted(set(files))


def zip_checkpoints(tag: str, seed: int, out_zip: Path = None) -> Path:
    """Bundle checkpoints into a downloadable zip file."""
    ckpt_dir, files = get_checkpoint_files(tag, seed)
    if not files:
        print(f"[!] No checkpoint files found to zip for {tag}_seed{seed}.")
        return None

    if out_zip is None:
        base_out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        out_zip = base_out / f"{tag}_seed{seed}_checkpoints.zip"

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"      BUNDLING CHECKPOINTS FOR MODEL {tag} (SEED {seed})")
    print("=" * 65)
    print(f"Directory: {ckpt_dir}\n")

    total_bytes = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            size_mb = f.stat().st_size / 1e6
            total_bytes += f.stat().st_size
            zf.write(f, arcname=f"{tag}_seed{seed}/{f.name}")
            print(f"  + Added: {f.name:<30} ({size_mb:6.2f} MB)")

    # Also create/update checkpoints_bundle.zip as standard bundle name
    if Path("/kaggle/working").exists():
        bundle_copy = Path("/kaggle/working/checkpoints_bundle.zip")
        if bundle_copy != out_zip:
            try:
                shutil.copy2(out_zip, bundle_copy)
            except Exception:
                pass

    zip_mb = out_zip.stat().st_size / 1e6
    print("-" * 65)
    print(f"Packed files       : {len(files)}")
    print(f"Uncompressed size  : {total_bytes / 1e6:.2f} MB")
    print(f"Compressed zip size: {zip_mb:.2f} MB")
    print(f"Destination        : {out_zip.resolve()}")
    print("=" * 65)
    print("[SUCCESS] Checkpoint bundle created successfully!")
    print("\nHow to download from Kaggle:")
    print(f"  1. In the Kaggle right-hand panel: expand 'Output'")
    print(f"  2. Click '{out_zip.name}' -> Download (three vertical dots)")
    return out_zip


def get_drive_service(creds_path: str = None):
    """Authenticate and build Google Drive API v3 client."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
    except ImportError:
        print("Installing google-api-python-client google-auth ...")
        os.system("pip install -q google-api-python-client google-auth")
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

    # Check for credentials in environment, secrets, or file
    creds_json = creds_path or os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    if not creds_json and Path("service_account.json").exists():
        creds_json = "service_account.json"
    if not creds_json and Path("/kaggle/working/service_account.json").exists():
        creds_json = "/kaggle/working/service_account.json"

    # Try loading from Kaggle Secrets if running in Kaggle
    if not creds_json:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            creds_json = user_secrets.get_secret("GDRIVE_SERVICE_ACCOUNT")
        except Exception:
            pass

    if not creds_json:
        return None

    scopes = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    if os.path.exists(str(creds_json)):
        creds = service_account.Credentials.from_service_account_file(str(creds_json), scopes=scopes)
    else:
        try:
            info = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        except Exception as e:
            print(f"[!] Error parsing service account credentials JSON: {e}")
            return None

    return build("drive", "v3", credentials=creds)


def find_or_create_subfolder(service, parent_folder_id: str, folder_name: str) -> str:
    """Find an existing folder in Google Drive or create it if missing."""
    query = f"name = '{folder_name}' and '{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res = service.files().list(q=query, fields="files(id, name)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder.get("id")


def upload_or_update_file(service, folder_id: str, file_path: Path):
    """Upload a file to Google Drive or update existing file in-place."""
    from googleapiclient.http import MediaFileUpload

    query = f"name = '{file_path.name}' and '{folder_id}' in parents and trashed = false"
    res = service.files().list(q=query, fields="files(id, name)").execute()
    existing = res.get("files", [])
    media = MediaFileUpload(str(file_path), resumable=True)

    if existing:
        file_id = existing[0]["id"]
        print(f"  Updating existing {file_path.name} in Google Drive (ID: {file_id})...")
        req = service.files().update(fileId=file_id, media_body=media, fields="id, name")
    else:
        print(f"  Uploading {file_path.name} to Google Drive...")
        metadata = {"name": file_path.name, "parents": [folder_id]}
        req = service.files().create(body=metadata, media_body=media, fields="id, name")

    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            print(f"    Progress: {int(status.progress() * 100)}%")

    return response.get("id")


def upload_checkpoints_to_drive(tag: str, seed: int, folder_id: str = None, creds_path: str = None):
    """Directly uploads all checkpoint files and zip bundle into Google Drive."""
    # 1. Check if running in Google Colab with mounted Drive
    colab_drive_dest = Path(f"/content/drive/MyDrive/mini_project/checkpoints/{tag}_seed{seed}")
    if colab_drive_dest.parent.parent.exists():
        print(f"[Colab] Google Drive mounted at /content/drive. Syncing directly...")
        colab_drive_dest.mkdir(parents=True, exist_ok=True)
        ckpt_dir, files = get_checkpoint_files(tag, seed)
        for f in files:
            shutil.copy2(f, colab_drive_dest / f.name)
            print(f"  Copied {f.name} to {colab_drive_dest}")
        print(f"✅ Successfully synced to Google Drive: {colab_drive_dest}")
        return True

    # 2. Check for folder ID
    fid = folder_id or os.environ.get("GDRIVE_CHECKPOINTS_FOLDER_ID")
    if not fid:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            fid = user_secrets.get_secret("GDRIVE_CHECKPOINTS_FOLDER_ID")
        except Exception:
            pass

    service = get_drive_service(creds_path)
    if not service:
        print("\n⚠️  Google Drive credentials not configured for automated headless upload.")
        print("To enable direct cloud upload into your Google Drive from Kaggle:")
        print("  1. In Google Cloud Console: Create a Service Account & download key JSON.")
        print("  2. In Google Drive: Share your 'checkpoints' folder with the Service Account email.")
        print("  3. In Kaggle: Add Kaggle Secret 'GDRIVE_SERVICE_ACCOUNT' with the JSON text,")
        print("     and 'GDRIVE_CHECKPOINTS_FOLDER_ID' with the Drive folder ID.")
        print("\nFallback: Checkpoint zip bundle is ready in /kaggle/working for 1-click download!")
        return False

    fid = clean_gdrive_id(fid)
    if not fid:
        print("[!] No Google Drive folder ID provided. Pass --folder-id <ID_OR_URL> or set GDRIVE_CHECKPOINTS_FOLDER_ID.")
        return False

    print(f"\nConnecting to Google Drive folder: {fid} ...")
    target_folder = find_or_create_subfolder(service, fid, f"{tag}_seed{seed}")

    ckpt_dir, files = get_checkpoint_files(tag, seed)
    zip_path = zip_checkpoints(tag, seed)
    if zip_path:
        files.append(zip_path)

    print(f"\nUploading {len(files)} files to Google Drive folder '{tag}_seed{seed}'...")
    for f in files:
        upload_or_update_file(service, target_folder, f)

    print(f"\n✅ All checkpoints for Model {tag} (seed {seed}) saved directly into Google Drive!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync checkpoints from Kaggle to Drive")
    parser.add_argument("action", choices=["zip", "upload"], help="zip checkpoints or upload to drive")
    parser.add_argument("--tag", default="B", help="model tag (e.g. B)")
    parser.add_argument("--seed", type=int, default=123, help="seed number")
    parser.add_argument("--folder-id", default=None, help="Google Drive folder ID to upload into")
    parser.add_argument("--creds", default=None, help="Path to Google Service Account JSON credentials")
    args = parser.parse_args()

    if args.action == "zip":
        zip_checkpoints(args.tag, args.seed)
    elif args.action == "upload":
        upload_checkpoints_to_drive(args.tag, args.seed, args.folder_id, args.creds)


if __name__ == "__main__":
    main()
