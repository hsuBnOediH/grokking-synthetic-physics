import argparse
import csv
import os
import sys
import time
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from HDF5PendulumDataset import make_splits
from models_conv import ConvBottleneckAE
from models import ContinuousBottleneckMAE
from models_dct import DCTBottleneckAE


def build_model(args, device):
    if args.model == "conv":
        model = ConvBottleneckAE(latent_dim=args.latent_dim, action_dim=2)
    elif args.model == "vit":
        model = ContinuousBottleneckMAE(latent_dim=args.latent_dim, action_dim=2)
    elif args.model == "dct":
        model = DCTBottleneckAE(latent_dim=args.latent_dim, action_dim=2)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model.to(device)


def save_reconstructions(pred, target, path, n=8):
    """Save a grid comparing predictions vs ground truth."""
    n = min(n, pred.shape[0])
    comparison = torch.stack([target[:n], pred[:n]], dim=1).reshape(-1, *pred.shape[1:])
    save_image(comparison, path, nrow=n, padding=2)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_z = []

    pbar = tqdm(loader, desc=f"Train ep{epoch}", file=sys.stdout,
                mininterval=30, dynamic_ncols=False, leave=False)
    for batch in pbar:
        s_t = batch["S_t"].to(device)
        s_t_next = batch["S_t_next"].to(device)
        action = batch["action"].to(device)

        pred_s_next, z_t = model(s_t, action)
        loss = criterion(pred_s_next, s_t_next)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_z.append(z_t.detach())
        pbar.set_postfix(loss=f"{total_loss/n_batches:.5f}")

    all_z = torch.cat(all_z, dim=0)
    return total_loss / n_batches, all_z


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="eval"):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_z = []
    last_pred = None
    last_target = None

    pbar = tqdm(loader, desc=f"Eval {split_name}", file=sys.stdout,
                mininterval=30, dynamic_ncols=False, leave=False)
    for batch in pbar:
        s_t = batch["S_t"].to(device)
        s_t_next = batch["S_t_next"].to(device)
        action = batch["action"].to(device)

        pred_s_next, z_t = model(s_t, action)
        loss = criterion(pred_s_next, s_t_next)

        total_loss += loss.item()
        n_batches += 1
        all_z.append(z_t)
        last_pred = pred_s_next
        last_target = s_t_next

    all_z = torch.cat(all_z, dim=0)
    return total_loss / n_batches, all_z, last_pred, last_target


def zstd_converged(history, patience, threshold):
    """True if z_std relative range over last `patience` epochs is below threshold."""
    if len(history) < patience:
        return False
    window = history[-patience:]
    hi, lo = max(window), min(window)
    return (hi - lo) / max(hi, 1e-9) < threshold


def save_checkpoint(path, epoch, model, optimizer, scheduler, train_loss,
                    iid_val_loss, zstd_history, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "iid_val_loss": iid_val_loss,
        "zstd_history": zstd_history,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train compression spectrum model")
    parser.add_argument("--model", default="conv", choices=["conv", "vit", "dct"])
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Max epochs (z_std early stopping may end sooner)")
    parser.add_argument("--h5_path", default="pendulum_data_v3.h5")
    parser.add_argument("--design_csv", default="episode_design.csv")
    parser.add_argument("--save_dir", default="runs/default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N epochs")
    # Resume
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint .pt to resume from. "
                             "If --epochs > original, resets lr for extension; "
                             "otherwise restores exact optimizer+scheduler state.")
    # Rolling checkpoint retention
    parser.add_argument("--keep_checkpoints", type=int, default=3,
                        help="Number of recent periodic checkpoints to keep (rolling)")
    # Z_std early stopping
    parser.add_argument("--min_epochs", type=int, default=200,
                        help="Minimum epochs before z_std early stopping is active")
    parser.add_argument("--zstd_patience", type=int, default=50,
                        help="Epoch window for z_std convergence check")
    parser.add_argument("--zstd_threshold", type=float, default=0.01,
                        help="Relative z_std range threshold to declare convergence (0.01 = 1%%)")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable z_std early stopping (run full --epochs)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"Device: {device} ({gpu_name}, {gpu_mem}GB)", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    loaders = make_splits(args.h5_path, args.design_csv, seed=args.seed)
    train_loader    = loaders["train"]
    val_loader      = loaders["iid_val"]
    near_ood_loader = loaders["near_ood"]
    far_ood_loader  = loaders["far_ood"]
    print(f"Splits: train={len(train_loader.dataset)} iid_val={len(val_loader.dataset)} "
          f"near_ood={len(near_ood_loader.dataset)} far_ood={len(far_ood_loader.dataset)}")

    model = build_model(args, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, latent_dim={args.latent_dim}, params={param_count/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch  = 1
    zstd_history = []
    ckpt_paths   = []   # rolling window of periodic checkpoint paths

    if args.resume:
        print(f"Resuming from {args.resume} ...", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        zstd_history = ckpt.get("zstd_history", [])
        original_epochs = ckpt.get("args", {}).get("epochs", args.epochs)

        if args.epochs > original_epochs:
            # Extension mode: original cosine schedule already decayed lr→0.
            # Reset to a fresh cosine over the remaining epochs so the model
            # can continue making progress.
            remaining = args.epochs - start_epoch + 1
            print(f"  Extension: {original_epochs}→{args.epochs} epochs. "
                  f"Fresh cosine schedule (lr={args.lr}) over {remaining} remaining epochs.",
                  flush=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining))
        else:
            # Crash-recovery mode: restore exact optimizer + scheduler state.
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=original_epochs)
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"  Crash recovery: epoch {start_epoch}/{args.epochs}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

    # CSV log — append when resuming, fresh otherwise
    log_path = os.path.join(args.save_dir, "log.csv")
    log_mode = "a" if (args.resume and os.path.exists(log_path)) else "w"
    log_file   = open(log_path, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if log_mode == "w":
        log_writer.writerow(["epoch", "train_loss", "iid_val_loss", "near_ood_loss",
                             "far_ood_loss", "z_mean", "z_std", "lr"])

    early_stop_str = (f"patience={args.zstd_patience}, threshold={args.zstd_threshold:.1%}, "
                      f"min_epochs={args.min_epochs}")
    print(f"\nTraining epoch {start_epoch}→{args.epochs} | "
          f"{'z_std early stop: ' + early_stop_str if not args.no_early_stop else 'no early stop'}")
    print("-" * 60)

    stop_reason = f"reached max epochs ({args.epochs})"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_z = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        iid_val_loss,  _, last_pred, last_target = evaluate(
            model, val_loader, criterion, device, "iid_val")
        near_ood_loss, _, _, _ = evaluate(
            model, near_ood_loader, criterion, device, "near_ood")
        far_ood_loss,  _, _, _ = evaluate(
            model, far_ood_loader, criterion, device, "far_ood")
        epoch_time = time.time() - t0
        scheduler.step()

        z_mean = train_z.mean().item()
        z_std  = train_z.std().item()
        lr     = optimizer.param_groups[0]["lr"]
        zstd_history.append(z_std)

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{iid_val_loss:.6f}",
                             f"{near_ood_loss:.6f}", f"{far_ood_loss:.6f}",
                             f"{z_mean:.4f}", f"{z_std:.4f}", f"{lr:.6f}"])
        log_file.flush()

        if epoch % 10 == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | iid_val={iid_val_loss:.6f} | "
                  f"near_ood={near_ood_loss:.6f} | far_ood={far_ood_loss:.6f} | "
                  f"z_std={z_std:.4f} | lr={lr:.6f} | {epoch_time:.1f}s", flush=True)

        # --- Periodic rolling checkpoint ---
        if epoch % args.save_every == 0 or epoch == start_epoch:
            img_path  = os.path.join(args.save_dir, "images", f"recon_epoch{epoch:04d}.png")
            save_reconstructions(last_pred.cpu(), last_target.cpu(), img_path)

            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler,
                            train_loss, iid_val_loss, zstd_history, args)
            ckpt_paths.append(ckpt_path)

            # Delete oldest checkpoint beyond the rolling window
            while len(ckpt_paths) > args.keep_checkpoints:
                old = ckpt_paths.pop(0)
                if os.path.exists(old):
                    os.remove(old)
                    print(f"  [rolling] Removed {old}", flush=True)

        # --- Z_std early stopping ---
        if not args.no_early_stop and epoch >= args.min_epochs:
            if zstd_converged(zstd_history, args.zstd_patience, args.zstd_threshold):
                stop_reason = (f"z_std converged (epoch {epoch}, "
                               f"patience={args.zstd_patience}, "
                               f"threshold={args.zstd_threshold:.1%})")
                print(f"\n[Early Stop] {stop_reason}", flush=True)
                # Ensure we have a checkpoint at the stop epoch
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt")
                if not os.path.exists(ckpt_path):
                    save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler,
                                    train_loss, iid_val_loss, zstd_history, args)
                break

    log_file.close()

    final_path = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Stop: {stop_reason}")
    print(f"Logs: {log_path}  Final model: {final_path}")


if __name__ == "__main__":
    main()
