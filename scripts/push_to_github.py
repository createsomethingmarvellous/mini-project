"""
push_to_github.py  –  Sync code from Google Drive → GitHub (Colab-friendly)
=============================================================================
Usage (in a Colab cell):
    !python scripts/push_to_github.py --message "checkpoint after epoch 8"

What it does:
  1. Configures git identity + token-based auth (no password prompt).
  2. Stages all changed/new files that are NOT in .gitignore.
  3. Commits with your message (or a timestamped default).
  4. Pushes to the configured remote branch.

Optional flags:
  --remote   <name>    Git remote to push to (default: origin)
  --branch   <name>    Branch to push to (default: current branch)
  --message  <msg>     Commit message (default: auto-timestamp)
  --dry-run            Print what would happen, do not actually push
  --also-pack-checkpoints
                       After the push, also compress the checkpoints/
                       directory into Drive/archives/checkpoints_latest.tar
                       (useful so Kaggle can re-download them easily)

Setup: before running this the first time, set these Colab Secrets
  (left sidebar → key icon):
    GITHUB_TOKEN   – a Personal Access Token with "repo" scope
    GITHUB_USER    – your GitHub username  (e.g. anup-raykar)
    GITHUB_EMAIL   – your commit email     (e.g. you@example.com)
    GITHUB_REPO    – full repo slug        (e.g. anup-raykar/mini_project)

If Colab Secrets are unavailable, the script falls back to environment
variables with the same names, or prompts interactively.
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command, print it, and optionally return output."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=False,
        capture_output=capture, text=True
    )
    if capture:
        return result
    if result.returncode != 0 and check:
        print(f"[ERROR] Command failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def get_secret(name: str, prompt_label: str | None = None) -> str:
    """
    Try to read a secret from Colab userdata, then env, then prompt.
    """
    # 1. Colab Secrets (userdata)
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(name)
        if val:
            return val
    except Exception:
        pass

    # 2. Environment variable
    val = os.environ.get(name, "")
    if val:
        return val

    # 3. Interactive prompt
    label = prompt_label or name
    import getpass
    val = getpass.getpass(f"Enter {label}: ")
    if not val:
        print(f"[ERROR] {name} is required.", file=sys.stderr)
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Push Drive code → GitHub from Colab/Kaggle")
    parser.add_argument("--remote",  default="origin", help="Git remote name (default: origin)")
    parser.add_argument("--branch",  default="",       help="Branch to push (default: current branch)")
    parser.add_argument("--message", default="",       help="Commit message")
    parser.add_argument("--dry-run", action="store_true", help="Print but do not execute git operations")
    parser.add_argument("--also-pack-checkpoints", action="store_true",
                        help="Also pack checkpoints/ into archives/checkpoints_latest.tar after pushing")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Locate repo root (the directory that contains .git/)
    # -----------------------------------------------------------------------
    # When called from any subfolder, walk up until we find .git
    repo_root = Path(__file__).resolve().parent.parent
    while not (repo_root / ".git").exists():
        if repo_root.parent == repo_root:
            print("[ERROR] Could not find .git directory. Are you inside the repo?", file=sys.stderr)
            sys.exit(1)
        repo_root = repo_root.parent

    print(f"\n{'='*60}")
    print(f"  push_to_github.py")
    print(f"  Repo root : {repo_root}")
    print(f"{'='*60}\n")

    # Change to repo root so all git commands work correctly
    os.chdir(repo_root)

    # -----------------------------------------------------------------------
    # Collect credentials
    # -----------------------------------------------------------------------
    token  = get_secret("GITHUB_TOKEN",  "GitHub Personal Access Token (repo scope)")
    user   = get_secret("GITHUB_USER",   "GitHub username")
    email  = get_secret("GITHUB_EMAIL",  "GitHub email")
    repo   = get_secret("GITHUB_REPO",   "GitHub repo slug (e.g. username/repo_name)")

    # -----------------------------------------------------------------------
    # Configure git identity
    # -----------------------------------------------------------------------
    print("─── Configuring git identity ───")
    if not args.dry_run:
        run(f'git config user.name  "{user}"')
        run(f'git config user.email "{email}"')

        # Inject token into remote URL so no password is needed
        # Format: https://<token>@github.com/<user>/<repo>.git
        remote_url = f"https://{token}@github.com/{repo}.git"
        run(f'git remote set-url {args.remote} "{remote_url}"')
    else:
        print(f"  [dry-run] would configure git as {user} <{email}>")
        print(f"  [dry-run] would set remote URL to https://***@github.com/{repo}.git")

    # -----------------------------------------------------------------------
    # Determine branch
    # -----------------------------------------------------------------------
    if args.branch:
        branch = args.branch
    else:
        result = run("git rev-parse --abbrev-ref HEAD", capture=True)
        branch = result.stdout.strip() if result.returncode == 0 else "main"
    print(f"\n─── Target branch: {branch} ───")

    # -----------------------------------------------------------------------
    # Show current git status
    # -----------------------------------------------------------------------
    print("\n─── Git status ───")
    run("git status --short")

    # -----------------------------------------------------------------------
    # Stage all non-ignored changes
    # -----------------------------------------------------------------------
    print("\n─── Staging changes ───")
    if not args.dry_run:
        run("git add -A")
    else:
        print("  [dry-run] would run: git add -A")

    # Check if there is anything to commit
    result = run("git diff --cached --name-only", capture=True)
    staged_files = [f for f in result.stdout.strip().splitlines() if f]
    if not staged_files:
        print("\n✅ Nothing to commit — working tree is clean.")
    else:
        print(f"\n  Files staged ({len(staged_files)}):")
        for f in staged_files:
            print(f"    + {f}")

        # -------------------------------------------------------------------
        # Commit
        # -------------------------------------------------------------------
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        message = args.message or f"auto: sync from Drive [{ts}]"
        print(f"\n─── Committing: '{message}' ───")
        if not args.dry_run:
            run(f'git commit -m "{message}"')
        else:
            print(f"  [dry-run] would commit: {message}")

        # -------------------------------------------------------------------
        # Push
        # -------------------------------------------------------------------
        print(f"\n─── Pushing to {args.remote}/{branch} ───")
        if not args.dry_run:
            run(f"git push {args.remote} {branch}")
            print(f"\n✅ Successfully pushed to {args.remote}/{branch}")
        else:
            print(f"  [dry-run] would push to {args.remote}/{branch}")

    # -----------------------------------------------------------------------
    # Optional: pack checkpoints into a tar archive on Drive
    # -----------------------------------------------------------------------
    if args.also_pack_checkpoints:
        print("\n─── Packing checkpoints ───")
        try:
            # Import config to get paths (may not be importable locally)
            sys.path.insert(0, str(repo_root / "scripts"))
            from config import CHECKPOINT_DIR, ARCHIVE_DIR  # type: ignore
            archive_path = ARCHIVE_DIR / "checkpoints_latest.tar"
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            if CHECKPOINT_DIR.exists():
                if not args.dry_run:
                    run(f'tar -cf "{archive_path}" -C "{CHECKPOINT_DIR.parent}" "{CHECKPOINT_DIR.name}"')
                    print(f"✅ Checkpoints packed → {archive_path}")
                else:
                    print(f"  [dry-run] would pack {CHECKPOINT_DIR} → {archive_path}")
            else:
                print(f"  ⚠️  Checkpoint dir not found: {CHECKPOINT_DIR}")
        except ImportError as e:
            print(f"  ⚠️  Could not import config.py: {e} — skipping checkpoint pack")

    print("\n─── Done ─────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
