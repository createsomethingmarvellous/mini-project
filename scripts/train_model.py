"""
Tasks T10 / T14 - train Model B (noise-augmented) or the clean control C.

Follows AASIST's own training recipe (same model, optimizer, cosine
schedule, loss weights, dev-based model selection and SWA), with three
project changes:
  1. Resumes after a Colab disconnect. Progress is saved every
     SAVE_EVERY_MINUTES (config.py), even in the middle of an epoch, so a
     disconnect loses at most that many minutes of training.
  2. Model selection uses ONLY the dev set. AASIST's main.py also picks
     "best.pth" using the eval set, which would leak test data into our
     results - we never do that. The final model is swa.pth.
  3. Extra training files (the augmented copies) can come from another folder.

Usage (in Colab):
    !python scripts/train_model.py --tag B --seed 42          # Model B
    !python scripts/train_model.py --tag C --seed 42 --clean  # clean control (only if epochs were cut)
Just run the same command again after a disconnect - it resumes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import protocol_path, read_protocol, split_dir
import aasist_bridge as ab
from utils import create_optimizer, seed_worker, set_seed  # AASIST utils


def training_files(clean_only: bool):
    rows = read_protocol(protocol_path("train"))
    train_dir = split_dir("train")
    train_flac = train_dir / "flac" if (train_dir / "flac").exists() else train_dir
    paths = {r[1]: train_flac / f"{r[1]}.flac" for r in rows}
    if not clean_only:
        aug_protocol = config.TRAIN_AUG_DIR / "protocol_aug.txt"
        if not aug_protocol.exists():
            cand_protocols = list(config.LOCAL_DATA.rglob("protocol_aug.txt"))
            if Path("/kaggle/input").exists():
                cand_protocols += list(Path("/kaggle/input").rglob("protocol_aug.txt"))
            if cand_protocols:
                aug_protocol = cand_protocols[0]
            else:
                sys.exit(f"ERROR: augmented training set missing ({aug_protocol}). Run T9 or unpack train_aug.tar first.")
        aug_rows = read_protocol(aug_protocol)
        aug_dir = aug_protocol.parent
        aug_flac = aug_dir / "flac" if (aug_dir / "flac").exists() else aug_dir
        have = set(os.listdir(aug_flac)) if aug_flac.exists() else set()
        missing = [r[1] for r in aug_rows if f"{r[1]}.flac" not in have]
        if missing:
            sys.exit(f"ERROR: {len(missing)} augmented files not built yet "
                     f"(e.g. {missing[0]}). Re-run T9 until it finishes.")
        rows += aug_rows
        paths.update({r[1]: aug_flac / f"{r[1]}.flac" for r in aug_rows})
    labels = {r[1]: int(r[4] == "bonafide") for r in rows}
    return paths, labels


def average_into(swa_state, model_state, n):
    """Running average of weights (what AASIST's SWA optimizer does)."""
    if swa_state is None:
        return {k: v.detach().clone() for k, v in model_state.items()}
    for k, v in model_state.items():
        if v.dtype.is_floating_point:
            swa_state[k].mul_(n / (n + 1)).add_(v.detach(), alpha=1 / (n + 1))
        else:
            swa_state[k] = v.detach().clone()
    return swa_state


class FocalLoss(nn.Module):
    """Focal Loss (gamma=2.0) down-weights easy samples and focuses gradients on hard noisy/enhanced clips."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=["B", "C"], required=True,
                        help="B = noise-augmented model, C = clean control")
    parser.add_argument("--seed", type=int, choices=config.SEEDS, required=True)
    parser.add_argument("--clean", action="store_true",
                        help="train on clean data only (use with --tag C)")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()
    if (args.tag == "C") != args.clean:
        sys.exit("Use --clean exactly when --tag C.")

    if not torch.cuda.is_available():
        sys.exit("No GPU - Runtime > Change runtime type > GPU.")
    device = "cuda"

    conf = ab.load_aasist_config()
    set_seed(args.seed, conf)
    run_dir = config.CHECKPOINT_DIR / f"{args.tag}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last_checkpoint.pth"

    # ---- data --------------------------------------------------------------
    paths, labels = training_files(args.clean)
    train_set = ab.TrainSet(paths, labels)
    bs = conf["batch_size"]
    batches_per_epoch = len(train_set) // bs      # drop_last, as in AASIST

    def epoch_loader(epoch, start_batch):
        """This epoch's shuffled batches, skipping the ones already trained.
        The shuffle order depends only on (seed, epoch), so after a resume
        the remaining batches are exactly the ones that were not done yet."""
        gen = torch.Generator()
        gen.manual_seed(args.seed * 1000 + epoch)
        order = torch.randperm(len(train_set), generator=gen).tolist()
        order = order[start_batch * bs: batches_per_epoch * bs]
        kwargs = {"pin_memory": True, "num_workers": args.workers,
                  "worker_init_fn": seed_worker, "generator": gen}
        if args.workers > 0:
            kwargs["prefetch_factor"] = 4
        return torch.utils.data.DataLoader(
            train_set, batch_size=bs, sampler=order, drop_last=True, **kwargs)

    dev_rows = read_protocol(protocol_path("dev"))
    dev_loader = ab.eval_loader(dev_rows, split_dir("dev"))
    if config.DATA_BASE != config.LOCAL_DATA:
        print("WARNING: reading audio from Drive (slow). Run "
              "'python scripts/fast_data.py unpack' first.")
    print(f"Training files: {len(paths)}  |  dev files: {len(dev_rows)}  |  "
          f"{batches_per_epoch} batches per epoch (batch size {bs})")
    if torch.cuda.is_available():
        res = torch.cuda.memory_reserved() / 1024**3
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU VRAM Active Tensors: {alloc:.1f} GB  |  GPU VRAM Reserved by CUDA: {res:.1f} GB (of 15.0 GB)")

    # ---- model / optimizer (AASIST recipe with Focal Loss + Frequency Masking) -----
    model = ab.build_model(conf["model_config"], device)
    optim_config = conf["optim_config"]
    optim_config["epochs"] = conf["num_epochs"]
    optim_config["steps_per_epoch"] = batches_per_epoch
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    criterion = FocalLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device), gamma=2.0)

    # epoch/batch = where training stopped; loss_sum/seen/minutes = this epoch so far
    state = {"epoch": 0, "batch": 0, "loss_sum": 0.0, "seen": 0, "minutes": 0.0,
             "best_dev_eer": 100.0, "swa": None, "n_swa": 0}
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        print(f"Resuming from epoch {state['epoch']}, "
              f"batch {state['batch']}/{batches_per_epoch}")

    def save_checkpoint():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, tmp)
        tmp.replace(ckpt_path)   # never leaves a half-written checkpoint

    # ---- training loop -----------------------------------------------------
    save_every = config.SAVE_EVERY_MINUTES * 60
    last_save = time.time()
    for epoch in range(state["epoch"], conf["num_epochs"]):
        start, minutes_before = time.time(), state["minutes"]
        model.train()
        bar = tqdm(epoch_loader(epoch, state["batch"]), desc=f"epoch {epoch}",
                   total=batches_per_epoch, initial=state["batch"],
                   unit="batch", mininterval=30)
        for batch_x, batch_y in bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(device)
            _, out = model(batch_x, Freq_aug=True)
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            state["batch"] += 1
            state["loss_sum"] += loss.item() * batch_x.size(0)
            state["seen"] += batch_x.size(0)

            if time.time() - last_save >= save_every:
                state["minutes"] = minutes_before + (time.time() - start) / 60
                save_checkpoint()
                last_save = time.time()
                tqdm.write(f"  saved: epoch {epoch}, batch {state['batch']}/{batches_per_epoch}")

        minutes = minutes_before + (time.time() - start) / 60
        bar.close()
        dev_eer = ab.write_scores(model, dev_loader, dev_rows,
                                  run_dir / "dev_scores.txt", device, progress=True)
        if dev_eer <= state["best_dev_eer"]:
            state["best_dev_eer"] = dev_eer
            state["swa"] = average_into(state["swa"], model.state_dict(), state["n_swa"])
            state["n_swa"] += 1
            torch.save(model.state_dict(), run_dir / "best_dev.pth")

        loss_avg = state["loss_sum"] / max(state["seen"], 1)
        state.update(epoch=epoch + 1, batch=0, loss_sum=0.0, seen=0, minutes=0.0)
        save_checkpoint()
        last_save = time.time()

        left = conf["num_epochs"] - epoch - 1
        line = (f"epoch {epoch:03d}  loss {loss_avg:.5f}  "
                f"dev EER {dev_eer:.3f}%  ({minutes:.1f} min training, "
                f"~{minutes * left / 60:.1f} h left for this seed)")
        print(line)
        with open(run_dir / "log.txt", "a") as f:
            f.write(line + "\n")

    # ---- final SWA model ---------------------------------------------------
    if not (run_dir / "swa.pth").exists():
        model.load_state_dict(state["swa"])
        bn_loader = torch.utils.data.DataLoader(train_set, batch_size=bs,
                                                num_workers=args.workers)
        print("Building the final SWA model (one pass over the training data) ...")
        torch.optim.swa_utils.update_bn(tqdm(bn_loader, desc="final SWA", unit="batch",
                                             mininterval=30), model, device=device)
        torch.save(model.state_dict(), run_dir / "swa.pth")
    print(f"Done. Final model: {run_dir / 'swa.pth'}")


if __name__ == "__main__":
    main()
