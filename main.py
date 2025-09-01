import time
from os.path import join
import os
import csv

import torch
from tqdm import tqdm

import Procedure
import register
import utils
import world
from register import dataset
from torch.utils.tensorboard import SummaryWriter

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
d = world.device
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best0_ndcg, best0_recall, best0_pre = 0, 0, 0
best1_ndcg, best1_recall, best1_pre = 0, 0, 0
best0_ndcg_cold, best0_recall_cold, best0_pre_cold = 0, 0, 0
best1_ndcg_cold, best1_recall_cold, best1_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0
start = time.time()

# ====== Logging setup (CSV + TensorBoard) ======
log_dir = os.path.join(world.BOARD_PATH, f"{world.model_name}_{world.dataset}")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_log.csv")
csv_file = None
csv_writer = None
if not os.path.exists(csv_path):
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "epoch",
            "loss_total",
            "loss_bpr",
            "loss_reg",
            "loss_attr",
            "loss_lbl",
            "ndcg@10",
            "ndcg@20",
            "recall@10",
            "recall@20",
            "precision@10",
            "precision@20",
        ]
    )
else:
    csv_file = open(csv_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated() / 1024 / 1024)
    return 0


tip = "pre"
try:
    # Create progress bar for training
    pbar = tqdm(range(world.TRAIN_epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:

        metrics = {
            "ndcg@10": None,
            "ndcg@20": None,
            "recall@10": None,
            "recall@20": None,
            "precision@10": None,
            "precision@20": None,
        }
        if epoch > 0 and (epoch % 50 == 0 or epoch == world.TRAIN_epochs):
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            # results_cold = Procedure.Test(dataset, Recmodel, epoch, True)
            if results["ndcg"][0] < best0_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best0_recall = results["recall"][0]
                best0_ndcg = results["ndcg"][0]
                best0_pre = results["precision"][0]
                low_count = 0

            if results["ndcg"][1] >= best1_ndcg:
                best1_recall = results["recall"][1]
                best1_ndcg = results["ndcg"][1]
                best1_pre = results["precision"][1]
            # populate metrics for logging
            metrics = {
                "ndcg@10": float(results["ndcg"][0]),
                "ndcg@20": float(results["ndcg"][1]),
                "recall@10": float(results["recall"][0]),
                "recall@20": float(results["recall"][1]),
                "precision@10": float(results["precision"][0]),
                "precision@20": float(results["precision"][1]),
            }
        loss_avg, comp_avgs = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch
        )

        # Update progress bar with current loss and GPU memory
        gpu_mem = get_gpu_memory()
        # Limit loss display to prevent progress bar cutoff
        loss_display = f"{loss_avg:.2f}" if loss_avg < 100 else f"{loss_avg:.1f}"
        pbar.set_postfix(loss=loss_display, gpu_mem=f"{gpu_mem}MB")

        # ===== Write to TensorBoard =====
        writer.add_scalar("loss/total", loss_avg, epoch)
        writer.add_scalar("loss/bpr", comp_avgs.get("bpr", 0.0), epoch)
        writer.add_scalar("loss/reg", comp_avgs.get("reg", 0.0), epoch)
        writer.add_scalar("loss/attr", comp_avgs.get("attr", 0.0), epoch)
        writer.add_scalar("loss/lbl", comp_avgs.get("lbl", 0.0), epoch)
        if metrics["ndcg@10"] is not None:
            writer.add_scalar("metric/ndcg@10", metrics["ndcg@10"], epoch)
            writer.add_scalar("metric/ndcg@20", metrics["ndcg@20"], epoch)
            writer.add_scalar("metric/recall@10", metrics["recall@10"], epoch)
            writer.add_scalar("metric/recall@20", metrics["recall@20"], epoch)
            writer.add_scalar("metric/precision@10", metrics["precision@10"], epoch)
            writer.add_scalar("metric/precision@20", metrics["precision@20"], epoch)

        # ===== Append to CSV =====
        csv_writer.writerow(
            [
                epoch,
                loss_avg,
                comp_avgs.get("bpr", 0.0),
                comp_avgs.get("reg", 0.0),
                comp_avgs.get("attr", 0.0),
                comp_avgs.get("lbl", 0.0),
                metrics["ndcg@10"],
                metrics["ndcg@20"],
                metrics["recall@10"],
                metrics["recall@20"],
                metrics["precision@10"],
                metrics["precision@20"],
            ]
        )
        csv_file.flush()

    pbar.close()
    end = time.time()
    print("The total time:", (end - start) / 60)
    # torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"best precision at 10:{best0_pre}")
    print(f"best precision at 20:{best1_pre}")
    print(f"best recall at 10:{best0_recall}")
    print(f"best recall at 20:{best1_recall}")
    print(f"best ndcg at 10:{best0_ndcg}")
    print(f"best ndcg at 20:{best1_ndcg}")
    try:
        csv_file and csv_file.close()
        writer and writer.flush()
        writer and writer.close()
    except Exception:
        pass
