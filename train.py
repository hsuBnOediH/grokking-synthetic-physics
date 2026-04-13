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


def build_model(args, device):
    if args.model == "conv":
        model = ConvBottleneckAE(latent_dim=args.latent_dim, action_dim=2)
    elif args.model == "vit":
        model = ContinuousBottleneckMAE(latent_dim=args.latent_dim, action_dim=2)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model.to(device)


def save_reconstructions(pred, target, path, n=8):
    """Save a grid comparing predictions vs ground truth."""
    n = min(n, pred.shape[0])
    # Interleave: pred1, target1, pred2, target2, ...
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


def main():
    parser = argparse.ArgumentParser(description="Train compression spectrum model")
    parser.add_argument("--model", default="conv", choices=["conv", "vit"])
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--h5_path", default="pendulum_data_v3.h5")
    parser.add_argument("--design_csv", default="episode_design.csv")
    parser.add_argument("--save_dir", default="runs/default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"Device: {device} ({gpu_name}, {gpu_mem}GB)", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    # Dataset
    loaders = make_splits(args.h5_path, args.design_csv, seed=args.seed)
    train_loader   = loaders["train"]
    val_loader     = loaders["iid_val"]
    near_ood_loader = loaders["near_ood"]
    far_ood_loader  = loaders["far_ood"]
    print(f"Splits: train={len(train_loader.dataset)} iid_val={len(val_loader.dataset)} "
          f"near_ood={len(near_ood_loader.dataset)} far_ood={len(far_ood_loader.dataset)}")

    # Model
    model = build_model(args, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, latent_dim={args.latent_dim}, params={param_count/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # CSV log
    log_path = os.path.join(args.save_dir, "log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "iid_val_loss", "near_ood_loss", "far_ood_loss", "z_mean", "z_std", "lr"])

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_z = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        iid_val_loss,  _,  last_pred, last_target = evaluate(model, val_loader,      criterion, device, "iid_val")
        near_ood_loss, _, _, _                    = evaluate(model, near_ood_loader, criterion, device, "near_ood")
        far_ood_loss,  _, _, _                    = evaluate(model, far_ood_loader,  criterion, device, "far_ood")
        epoch_time = time.time() - t0
        scheduler.step()

        z_mean = train_z.mean().item()
        z_std = train_z.std().item()
        lr = optimizer.param_groups[0]["lr"]

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{iid_val_loss:.6f}",
                             f"{near_ood_loss:.6f}", f"{far_ood_loss:.6f}",
                             f"{z_mean:.4f}", f"{z_std:.4f}", f"{lr:.6f}"])
        log_file.flush()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | iid_val={iid_val_loss:.6f} | "
                  f"near_ood={near_ood_loss:.6f} | far_ood={far_ood_loss:.6f} | "
                  f"z_std={z_std:.4f} | lr={lr:.6f} | {epoch_time:.1f}s", flush=True)

        # Save reconstruction images
        if epoch % args.save_every == 0 or epoch == 1:
            img_path = os.path.join(args.save_dir, "images", f"recon_epoch{epoch:04d}.png")
            save_reconstructions(last_pred.cpu(), last_target.cpu(), img_path)

            # Save checkpoint
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "iid_val_loss": iid_val_loss,
                "args": vars(args),
            }, ckpt_path)

    log_file.close()

    # Save final model
    final_path = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Logs: {log_path}, Final model: {final_path}")


if __name__ == "__main__":
    main()
