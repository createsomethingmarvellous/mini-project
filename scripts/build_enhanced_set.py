"""
Task T12 - build the enhanced (cleaned) test sets.

Runs a PRETRAINED speech enhancement model over each noisy test set from
T6. Nothing is trained here; SpeechBrain downloads the weights once.

    generated/eval/noisy_snr{X}/flac  ->  generated/eval/{model}_snr{X}/flac

Finished files are skipped, so it is safe to re-run after a disconnect.

Usage (in Colab) - try --limit 20 first:
    !python scripts/build_enhanced_set.py --model metricgan --limit 20
    !python scripts/build_enhanced_set.py --model metricgan
    !python scripts/build_enhanced_set.py --model sepformer
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import eval_rows_for, load_audio, save_flac

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.ENHANCEMENT_MODELS),
                        required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only N files per SNR - test with this first")
    parser.add_argument("--batch_size", type=int, default=config.ENHANCEMENT_BATCH_SIZE,
                        help=f"batch size for GPU inference (default: {config.ENHANCEMENT_BATCH_SIZE})")
    args = parser.parse_args()

    rows = eval_rows_for("noisy")[:args.limit]
    enhance = load_enhancer(args.model)

    for snr in config.EVAL_SNRS_DB:
        src_dir = config.EVAL_CONDITIONS_DIR / f"noisy_snr{snr}" / "flac"
        out_dir = config.EVAL_CONDITIONS_DIR / f"{args.model}_snr{snr}" / "flac"
        if not src_dir.exists():
            sys.exit(f"ERROR: {src_dir} missing. Run T6 first: "
                     "python scripts/build_noisy_set.py --mode eval")

        todo = []
        skipped = 0
        for row in rows:
            out = out_dir / f"{row[1]}.flac"
            if out.exists():
                skipped += 1
            else:
                todo.append((src_dir / f"{row[1]}.flac", out))

        failed = 0
        written = 0
        bs = args.batch_size

        pbar = tqdm(total=len(rows), desc=f"{args.model}_snr{snr}", unit="file", mininterval=5)
        pbar.update(skipped)

        for i in range(0, len(todo), bs):
            chunk = todo[i : i + bs]
            try:
                wavs = [load_audio(src) for src, _ in chunk]
                max_len = max(len(w) for w in wavs)
                batch_tensor = torch.zeros(len(wavs), max_len, dtype=torch.float32)
                lengths_tensor = torch.tensor([len(w) / max_len for w in wavs], dtype=torch.float32)
                for idx, w in enumerate(wavs):
                    batch_tensor[idx, :len(w)] = torch.from_numpy(w)

                with torch.no_grad():
                    enhanced_batch = enhance(batch_tensor, lengths_tensor)

                for idx, (src, out) in enumerate(chunk):
                    orig_len = len(wavs[idx])
                    save_flac(out, enhanced_batch[idx, :orig_len].cpu().numpy())
                    written += 1
                    pbar.update(1)

            except Exception as e:
                # Fallback to single-file processing if batch fails
                for src, out in chunk:
                    try:
                        w = load_audio(src)
                        b_t = torch.from_numpy(w).unsqueeze(0)
                        l_t = torch.tensor([1.0], dtype=torch.float32)
                        with torch.no_grad():
                            enh = enhance(b_t, l_t)[0]
                        save_flac(out, enh.cpu().numpy())
                        written += 1
                    except Exception as single_e:
                        failed += 1
                        if failed <= 5:
                            print(f"\n  failed on {src.name}: {single_e}")
                    pbar.update(1)

        pbar.close()
        print(f"{args.model}_snr{snr}: {written} written, {skipped} already existed, {failed} failed")
        if failed:
            print("WARNING: investigate failures before continuing.")


if __name__ == "__main__":
    main()

