"""
predict.py  –  COL761 Assignment 3 prediction script
"""

import argparse
import os

import numpy as np
import torch

from Q2.src.load_dataset import COL761LinkDataset, COL761NodeDataset, _load_edge_list, load_dataset


def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    return model


def _random_A(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))


def _random_B(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)


def _random_C(V: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(V), torch.rand(V, K)


@torch.no_grad()
def predict_A(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    data = dataset[0]
    logits = model(data.x, data.edge_index)
    return logits.argmax(dim=1)


@torch.no_grad()
def predict_B(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    data = dataset[0]
    logits = model(data.x, data.edge_index)
    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1)
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def predict_C(
    model: torch.nn.Module,
    dataset: COL761LinkDataset,
    test_dir: str = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if test_dir is None:
        pos = dataset.valid_pos
        neg = dataset.valid_neg
        split = "valid"
    else:
        pos = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))
        npy = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))
        split = "test"

    P, K, _ = neg.shape
    pos_scores = model(dataset.x, dataset.edge_index, pos)
    neg_scores = model(dataset.x, dataset.edge_index, neg.view(P * K, 2)).view(P, K)
    return pos_scores, neg_scores, split


def predict_and_save(
    dataset_name: str,
    data_dir: str,
    model_path: str,
    out_dir: str,
    test_dir: str = None,
    kerberos: str = "student",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        model = load_model(model_path)
    else:
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred}, out_path)
        return

    if dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score}, out_path)
        return

    if model:
        pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
    else:
        if test_dir or not hasattr(ds, "valid_pos"):
            pos = ds.test_pos
            neg = ds.test_neg
            split = "test"
        else:
            pos = ds.valid_pos
            neg = ds.valid_neg
            split = "valid"
        V, K = pos.shape[0], neg.shape[1]
        pos_scores, neg_scores = _random_C(V, K)

    out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
    torch.save({"pos_scores": pos_scores, "neg_scores": neg_scores, "split": split}, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for COL761 A3 datasets.")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--test_dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")

    predict_and_save(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_path=model_path,
        out_dir=args.output_dir,
        test_dir=args.test_dir,
        kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()
