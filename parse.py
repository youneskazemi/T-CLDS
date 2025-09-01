import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # for hyper parameters
    parser.add_argument("-m", "--model", type=str, default="CLDS")
    parser.add_argument("-d", "--dataset", type=str, default="douban")
    parser.add_argument("--recdim", type=int, default=64, help="the embedding size")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-4,
        help="the weight decay for l2 normalization",
    )
    parser.add_argument(
        "--bpr_batch",
        type=int,
        default=2048,
        help="the batch size for bpr loss training procedure",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    # for deep model
    parser.add_argument("--layer", type=int, default=3, help="the layer num of graphs")
    # normally unchanged
    parser.add_argument("--topks", nargs="?", default="[10, 20]", help="@k test list")
    parser.add_argument(
        "--testbatch", type=str, default=100, help="the batch size of users for testing"
    )
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2020, help="random seed")

    # ===== Temporal encoder & time options =====
    parser.add_argument(
        "--use_temporal", action="store_true", help="Enable temporal-aware encoder"
    )
    parser.add_argument(
        "--temporal_mode",
        type=str,
        default="t2v",
        choices=["t2v", "pe"],
        help="Time feature type: Time2Vec ('t2v') or Fourier positional ('pe')",
    )
    parser.add_argument(
        "--time_feat_dim",
        type=int,
        default=32,
        help="Temporal feature hidden size before fusion",
    )
    parser.add_argument(
        "--time_tau",
        type=float,
        default=24.0,
        help="Temporal decay horizon (in hours) for recency weighting",
    )
    parser.add_argument(
        "--time_unit",
        type=str,
        default="hours",
        choices=["seconds", "minutes", "hours", "days"],
        help="Unit of raw timestamps for decay weighting",
    )
    parser.add_argument(
        "--time_norm",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "log1p", "none"],
        help="How to normalize timestamps fed into the temporal encoder",
    )

    return parser.parse_args()
