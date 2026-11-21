"""
Task T12 - build the enhanced (cleaned) test sets and training sets.

Runs a PRETRAINED speech enhancement model over:
  - eval mode: each noisy test set from T6 (noisy_snr0, 10, 20)
  - train mode: noise-augmented training clips for Model D (enhancement-aware training)

Finished files are skipped, so it is safe to re-run after a disconnect.

Usage:
    # 1. Test sets:
    !python scripts/build_enhanced_set.py --mode eval --model metricgan --amp
    !python scripts/build_enhanced_set.py --mode eval --model sepformer --amp

    # 2. Training set for Model D (enhancement-aware):
    !python scripts/build_enhanced_set.py --mode train --model metricgan --amp
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import (eval_rows_for, find_file_fast, load_audio,
                    read_protocol, save_flac, write_protocol)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_enhancer(model_key: str):
    """Return a function: (batch_wavs [B, T], lengths [B]) -> batch_enhanced_tensor [B, T]."""
    try:
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError:     # older SpeechBrain versions
        from speechbrain.pretrained import (SepformerSeparation,
                                            SpectralMaskEnhancement)

    source = config.ENHANCEMENT_MODELS[model_key]
    savedir = str(config.BASE / "pretrained_enhance" / model_key)
    run_opts = {"device": DEVICE}
    print(f"Loading {model_key} from {source} on {DEVICE} ...")

    if model_key == "metricgan":
        model = SpectralMaskEnhancement.from_hparams(
            source=source, savedir=savedir, run_opts=run_opts)
        return lambda wavs, lengths: model.enhance_batch(
            wavs.to(DEVICE), lengths=lengths.to(DEVICE))

    if model_key == "sepformer":
        model = SepformerSeparation.from_hparams(
            source=source, savedir=savedir, run_opts=run_opts)
        # output shape [batch, time, n_sources]; this model has 1 source
        return lambda wavs, lengths: model.separate_batch(
            wavs.to(DEVICE))[:, :, 0]

    sys.exit(f"No loader written for {model_key}")


def run_batch_enhancement(todo, enhance, bs, amp, desc):
    failed = 0
    written = 0
    pbar = tqdm(total=len(todo), desc=desc, unit="file", mininterval=5)

    for i in range(0, len(todo), bs):
        chunk = todo[i : i + bs]
        try:
            wavs = [load_audio(src) for src, _ in chunk]
            max_len = max(len(w) for w in wavs)
            batch_tensor = torch.zeros(len(wavs), max_len, dtype=torch.float32)
            lengths_tensor = torch.tensor([len(w) / max_len for w in wavs], dtype=torch.float32)
            for idx, w in enumerate(wavs):
                batch_tensor[idx, :len(w)] = torch.from_numpy(w)

            try:
                autocast_ctx = torch.amp.autocast('cuda', enabled=amp)
            except (TypeError, AttributeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

            with torch.no_grad():
                with autocast_ctx:
                    enhanced_batch = enhance(batch_tensor, lengths_tensor)

            for idx, (src, out) in enumerate(chunk):
                orig_len = len(wavs[idx])
                save_flac(out, enhanced_batch[idx, :orig_len].cpu().float().numpy())
                written += 1
                pbar.update(1)

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for src, out in chunk:
                try:
                    w = load_audio(src)
                    b_t = torch.from_numpy(w).unsqueeze(0)
                    l_t = torch.tensor([1.0], dtype=torch.float32)
                    with torch.no_grad():
                        with autocast_ctx:
                            enh = enhance(b_t, l_t)[0]
                    save_flac(out, enh.cpu().float().numpy())
                    written += 1
                except Exception as single_e:
                    failed += 1
                    if failed <= 5:
                        print(f"\n  failed on {src.name}: {single_e}")
                pbar.update(1)

    pbar.close()
    return written, failed


def build_train_mode(args, enhance):
    search_bases = [config.LOCAL_DATA, config.TRAIN_AUG_DIR]
    if Path("/kaggle/input").exists():
        search_bases.append(Path("/kaggle/input"))
    aug_proto = find_file_fast("protocol_aug.txt", search_bases) or (config.TRAIN_AUG_DIR / "protocol_aug.txt")
    if not aug_proto.exists():
        sys.exit(f"ERROR: protocol_aug.txt not found ({aug_proto}). Build train_aug first.")

    aug_rows = read_protocol(aug_proto)
    src_flac_dir = aug_proto.parent / "flac" if (aug_proto.parent / "flac").exists() else aug_proto.parent
    out_dir = config.TRAIN_ENHANCED_DIR / "flac"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enhance noise and noise+reverb samples (_n and _nr)
    target_rows = [r for r in aug_rows if r[1].endswith("_n") or r[1].endswith("_nr")]
    if args.limit:
        target_rows = target_rows[:args.limit]

    print(f"Enhancing {len(target_rows)} training samples with {args.model} -> {out_dir}")
    todo = []
    skipped = 0
    enh_rows = []
    for r in target_rows:
        key_enh = f"{r[1]}_enh"
        out_path = out_dir / f"{key_enh}.flac"
        enh_rows.append([r[0], key_enh, r[2], r[3], r[4]])
        if out_path.exists():
            skipped += 1
        else:
            todo.append((src_flac_dir / f"{r[1]}.flac", out_path))

    write_protocol(config.TRAIN_ENHANCED_DIR / "protocol_enhanced.txt", enh_rows)
    print(f"protocol_enhanced.txt written with {len(enh_rows)} entries ({skipped} already exist on disk).")

    bs = args.batch_size
    if args.model == "sepformer" and bs > 8:
        bs = 8
    written, failed = run_batch_enhancement(todo, enhance, bs, args.amp, f"train_enhanced_{args.model}")
    print(f"train_enhanced: {written} written, {skipped} already existed, {failed} failed")


def build_eval_mode(args, enhance):
    rows = eval_rows_for("noisy")[:args.limit]
    for snr in config.EVAL_SNRS_DB:
        src_dir = config.EVAL_CONDITIONS_DIR / f"noisy_snr{snr}" / "flac"
        out_dir = config.EVAL_CONDITIONS_DIR / f"{args.model}_snr{snr}" / "flac"
        if not src_dir.exists():
            sys.exit(f"ERROR: {src_dir} missing. Run T6 first.")

        todo = []
        skipped = 0
        for row in rows:
            out = out_dir / f"{row[1]}.flac"
            if out.exists():
                skipped += 1
            else:
                todo.append((src_dir / f"{row[1]}.flac", out))

        bs = args.batch_size
        if args.model == "sepformer" and bs > 8:
            bs = 8

        print(f"Processing {len(todo)} files for {args.model}_snr{snr} ({skipped} skipped)...")
        written, failed = run_batch_enhancement(todo, enhance, bs, args.amp, f"{args.model}_snr{snr}")
        print(f"{args.model}_snr{snr}: {written} written, {skipped} already existed, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.ENHANCEMENT_MODELS), required=True)
    parser.add_argument("--mode", choices=["eval", "train"], default="eval",
                        help="eval = noisy test sets (default), train = train_aug audio for Model D")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only N files - test with this first")
    parser.add_argument("--batch_size", type=int, default=config.ENHANCEMENT_BATCH_SIZE,
                        help=f"batch size for GPU inference (default: {config.ENHANCEMENT_BATCH_SIZE})")
    parser.add_argument("--amp", action="store_true",
                        help="enable FP16 Automatic Mixed Precision for faster inference")
    args = parser.parse_args()

    enhance = load_enhancer(args.model)
    if args.mode == "train":
        build_train_mode(args, enhance)
    else:
        build_eval_mode(args, enhance)


if __name__ == "__main__":
    main()
