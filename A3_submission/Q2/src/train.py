"""
train.py — unified entry point matching the Piazza-specified train command:

    python train.py --dataset A|B|C --task node|link \
        --data_dir /abs/path/to/datasets \
        --model_dir /abs/path/to/models \
        --kerberos YOUR_KERBEROS

Saves the trained model to: <model_dir>/<kerberos>_model_<dataset>.pt
This filename is what predict.py loads (see predict.py: --model_dir).
"""

import argparse
import os
import sys


def _validate_task_dataset(dataset: str, task: str, parser: argparse.ArgumentParser) -> None:
    valid = {"node": ("A", "B"), "link": ("C",)}
    if dataset not in valid[task]:
        parser.error(
            f"--task {task} is not valid for --dataset {dataset}. "
            f"Expected dataset in {valid[task]}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entry point for COL761 A3 Q2.")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    args = parser.parse_args()

    _validate_task_dataset(args.dataset, args.task, parser)

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    os.makedirs(args.model_dir, exist_ok=True)

    forwarded = [
        sys.argv[0],
        "--data_dir", args.data_dir,
        "--out_dir", args.model_dir,
        "--kerberos", args.kerberos,
    ]

    if args.dataset == "A":
        from train_A import main as run
    elif args.dataset == "B":
        from train_B import main as run
        forwarded += ["--models", "pmlp"]
    else:
        from train_C import main as run

    sys.argv = forwarded
    run()

    expected = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")
    if not os.path.isfile(expected):
        raise FileNotFoundError(
            f"Training finished but expected model not found at {expected}. "
            f"predict.py will not be able to locate it."
        )


if __name__ == "__main__":
    main()
