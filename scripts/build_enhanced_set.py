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
    """Return a function: 1-D float waveform tensor -> 1-D enhanced tensor."""
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
        return lambda wav: model.enhance_batch(
            wav.unsqueeze(0).to(DEVICE),
            lengths=torch.tensor([1.0], device=DEVICE))[0]

    if model_key == "sepformer":
        model = SepformerSeparation.from_hparams(
            source=source, savedir=savedir, run_opts=run_opts)
        # output shape [batch, time, n_sources]; this model has 1 source
        return lambda wav: model.separate_batch(
            wav.unsqueeze(0).to(DEVICE))[0, :, 0]

    sys.exit(f"No loader written for {model_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(config.ENHANCEMENT_MODELS),
                        required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only N files per SNR - test with this first")
    args = parser.parse_args()

    rows = eval_rows_for("noisy")[:args.limit]
    enhance = load_enhancer(args.model)

    for snr in config.EVAL_SNRS_DB:
        src_dir = config.EVAL_CONDITIONS_DIR / f"noisy_snr{snr}" / "flac"
        out_dir = config.EVAL_CONDITIONS_DIR / f"{args.model}_snr{snr}" / "flac"
        if not src_dir.exists():
            sys.exit(f"ERROR: {src_dir} missing. Run T6 first: "
                     "python scripts/build_noisy_set.py --mode eval")

        failed = skipped = 0
        for row in tqdm(rows, desc=f"{args.model}_snr{snr}", unit="file", mininterval=5):
            out = out_dir / f"{row[1]}.flac"
            if out.exists():
                skipped += 1
                continue
            try:
                wav = torch.from_numpy(load_audio(src_dir / f"{row[1]}.flac"))
                with torch.no_grad():
                    enhanced = enhance(wav)
                save_flac(out, enhanced.cpu().numpy())
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\n  failed on {row[1]}: {e}")

        print(f"{args.model}_snr{snr}: {len(rows) - failed - skipped} written, "
              f"{skipped} already existed, {failed} failed")
        if failed:
            print("WARNING: investigate failures before continuing.")


if __name__ == "__main__":
    main()
