"""
Compare saved activation files with CKA.

This adapts the official "Similarity of Neural Network Representations
Revisited" demo implementation to the activation files produced by
scripts/capture_activations.py.

Typical usage:
    python scripts/cka_compare_activations.py \
        --act_a combined_models/act_task1.pt \
        --act_b combined_models/act_task2.pt

Outputs default to:
    scripts/cka/cka_<act_a_stem>_vs_<act_b_stem>.json
    scripts/cka/cka_<act_a_stem>_vs_<act_b_stem>.csv
    scripts/cka/cka_<act_a_stem>_vs_<act_b_stem>.png

Notes:
    CKA is most interpretable when both activation files were captured on the
    same ordered observation set. It can still be used as a rough distributional
    comparison for own-policy rollouts, but then rows are not paired examples.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


DEFAULT_KEYS = [
    "obs_out",
    "qkv.0",
    "qkv.0_out",
    "v.0",
    "v.0_out",
    "ffn1.0",
    "ffn1.0_out",
    "ffn2.0",
    "ffn2.0_out",
    "qkv.1",
    "qkv.1_out",
    "v.1",
    "v.1_out",
    "ffn1.1",
    "ffn1.1_out",
    "ffn2.1",
    "ffn2.1_out",
]


# Official demo implementation, kept close to the source.
def gram_linear(x):
    """Compute Gram matrix for a linear kernel."""
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram matrix for an RBF kernel using median distance bandwidth."""
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    if sq_median_distance <= 0:
        sq_median_distance = 1e-12
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix."""
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()

    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA from two Gram matrices."""
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    denom = normalization_x * normalization_y
    return float(scaled_hsic / denom) if denom > 0 else float("nan")


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity."""
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute linear CKA in feature space."""
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        if n <= 2:
            return float("nan")
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            max(
                _debiased_dot_product_similarity_helper(
                    normalization_x ** 2,
                    sum_squared_rows_x,
                    sum_squared_rows_x,
                    squared_norm_x,
                    squared_norm_x,
                    n,
                ),
                0.0,
            )
        )
        normalization_y = np.sqrt(
            max(
                _debiased_dot_product_similarity_helper(
                    normalization_y ** 2,
                    sum_squared_rows_y,
                    sum_squared_rows_y,
                    squared_norm_y,
                    squared_norm_y,
                    n,
                ),
                0.0,
            )
        )

    denom = normalization_x * normalization_y
    return float(dot_product_similarity / denom) if denom > 0 else float("nan")


def cca(features_x, features_y):
    """Compute mean squared CCA correlation from the official demo."""
    qx, _ = np.linalg.qr(features_x)
    qy, _ = np.linalg.qr(features_y)
    denom = min(features_x.shape[1], features_y.shape[1])
    return float(np.linalg.norm(qx.T.dot(qy)) ** 2 / denom) if denom > 0 else float("nan")


def load_activations(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}


def tensor_to_matrix(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    if x.dim() == 1:
        x = x[:, None]
    elif x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.numpy().astype(np.float64, copy=False)


def choose_keys(act_a: Dict[str, torch.Tensor], act_b: Dict[str, torch.Tensor], requested: str) -> List[str]:
    common = [k for k in DEFAULT_KEYS if k in act_a and k in act_b]
    if requested == "all":
        extras = sorted(k for k in set(act_a).intersection(act_b) if k not in common and k != "obs")
        return common + extras
    if requested == "default":
        return common
    keys = [k.strip() for k in requested.split(",") if k.strip()]
    missing = [k for k in keys if k not in act_a or k not in act_b]
    if missing:
        raise KeyError(f"Requested activation keys missing from one file: {missing}")
    return keys


def paired_sample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        x = x[idx]
        y = y[idx]
    return x, y


def similarity(x: np.ndarray, y: np.ndarray, metric: str, debiased: bool, rbf_threshold: float) -> float:
    if metric == "linear":
        return feature_space_linear_cka(x, y, debiased=debiased)
    if metric == "rbf":
        return cka(gram_rbf(x, rbf_threshold), gram_rbf(y, rbf_threshold), debiased=debiased)
    if metric == "cca":
        return cca(x - x.mean(0, keepdims=True), y - y.mean(0, keepdims=True))
    raise ValueError(metric)


def compute_matrix(
    act_a: Dict[str, torch.Tensor],
    act_b: Dict[str, torch.Tensor],
    keys_a: Iterable[str],
    keys_b: Iterable[str],
    metric: str,
    max_samples: int,
    seed: int,
    debiased: bool,
    rbf_threshold: float,
):
    rows = []
    matrices_a = {k: tensor_to_matrix(act_a[k]) for k in keys_a}
    matrices_b = {k: tensor_to_matrix(act_b[k]) for k in keys_b}

    for ka, xa0 in matrices_a.items():
        for kb, xb0 in matrices_b.items():
            xa, xb = paired_sample(xa0, xb0, max_samples=max_samples, seed=seed)
            value = similarity(xa, xb, metric=metric, debiased=debiased, rbf_threshold=rbf_threshold)
            rows.append(
                {
                    "layer_a": ka,
                    "layer_b": kb,
                    "metric": metric,
                    "value": value,
                    "n_examples": int(xa.shape[0]),
                    "dim_a": int(xa.shape[1]),
                    "dim_b": int(xb.shape[1]),
                }
            )
    return rows


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path: str, rows: List[dict], keys_a: List[str], keys_b: List[str], title: str) -> None:
    import matplotlib.pyplot as plt

    values = np.full((len(keys_a), len(keys_b)), np.nan, dtype=np.float64)
    ia = {k: i for i, k in enumerate(keys_a)}
    ib = {k: i for i, k in enumerate(keys_b)}
    for row in rows:
        values[ia[row["layer_a"]], ib[row["layer_b"]]] = row["value"]

    fig_w = max(8, 0.55 * len(keys_b))
    fig_h = max(6, 0.45 * len(keys_a))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(keys_b)), keys_b, rotation=60, ha="right")
    ax.set_yticks(range(len(keys_a)), keys_a)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_same_layer(rows: List[dict]) -> None:
    same = [r for r in rows if r["layer_a"] == r["layer_b"]]
    same.sort(key=lambda r: r["value"])
    print("\nSame-layer representation similarity:")
    print(f"{'Layer':<16} {'metric':<8} {'value':>8} {'N':>8} {'dim':>10}")
    print("-" * 58)
    for r in same:
        print(
            f"{r['layer_a']:<16} {r['metric']:<8} {r['value']:>8.4f} "
            f"{r['n_examples']:>8} {r['dim_a']:>5}/{r['dim_b']:<4}"
        )


def default_output_paths(act_a: str, act_b: str, metric: str) -> Tuple[str, str, str]:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "cka"
    stem_a = Path(act_a).stem
    stem_b = Path(act_b).stem
    base = f"cka_{stem_a}_vs_{stem_b}_{metric}"
    return (
        str(output_dir / f"{base}.json"),
        str(output_dir / f"{base}.csv"),
        str(output_dir / f"{base}.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CKA/SVCCA-style activation comparison for captured drone activations.")
    parser.add_argument("--act_a", required=True, help="Task/model A activation .pt file")
    parser.add_argument("--act_b", required=True, help="Task/model B activation .pt file")
    parser.add_argument("--keys", default="default", help="'default', 'all', or comma-separated activation keys")
    parser.add_argument("--metric", choices=["linear", "rbf", "cca"], default="linear")
    parser.add_argument("--debiased", action="store_true", help="Use debiased CKA estimator")
    parser.add_argument("--rbf-threshold", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=4096, help="0 means use all paired rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="JSON output path. Default: scripts/cka/")
    parser.add_argument("--csv", default=None, help="CSV output path. Default: scripts/cka/")
    parser.add_argument("--plot", default=None, help="Heatmap PNG path. Default: scripts/cka/")
    args = parser.parse_args()

    default_json, default_csv, default_plot = default_output_paths(args.act_a, args.act_b, args.metric)
    args.output = args.output or default_json
    args.csv = args.csv or default_csv
    args.plot = args.plot or default_plot

    act_a = load_activations(args.act_a)
    act_b = load_activations(args.act_b)
    keys = choose_keys(act_a, act_b, args.keys)
    if not keys:
        raise RuntimeError("No common activation keys found.")

    rows = compute_matrix(
        act_a,
        act_b,
        keys,
        keys,
        metric=args.metric,
        max_samples=args.max_samples,
        seed=args.seed,
        debiased=args.debiased,
        rbf_threshold=args.rbf_threshold,
    )

    print(f"\nLoaded A: {args.act_a}")
    print(f"Loaded B: {args.act_b}")
    print(f"Metric : {args.metric}{' (debiased)' if args.debiased else ''}")
    print(f"Keys   : {len(keys)}")
    print_same_layer(rows)

    result = {
        "act_a": args.act_a,
        "act_b": args.act_b,
        "metric": args.metric,
        "debiased": args.debiased,
        "rbf_threshold": args.rbf_threshold,
        "max_samples": args.max_samples,
        "keys": keys,
        "rows": rows,
        "note": (
            "CKA is most interpretable when rows are the same ordered examples. "
            "Own-policy activation files are useful for rough distributional comparison, "
            "but shared-observation captures are cleaner for mechanistic claims."
        ),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Wrote JSON: {args.output}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, rows)
        print(f"[OK] Wrote CSV:  {args.csv}")
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_heatmap(args.plot, rows, keys, keys, f"{args.metric.upper()} similarity")
        print(f"[OK] Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
