"""Compare battery/BS input pathway across merge parents.

Hypothesis: task1_v9 (trained WITH live battery/BS obs) actively learned those
input columns in a navigation-serving direction that CONFLICTS with task2_v5's
charge-timing direction, so blending cancels them -> unstable v4 soups. task1_v7
(battery=const 1.0, BS=0 during training) left those columns ~inert, so v2 blends
didn't fight task2.

Tests:
  (1) input_proj column norm for the battery/BS dims: v7 (inert) vs v9 (trained).
  (2) cosine(task1_col, task2_col) for those dims: v9 should be MORE anti-aligned
      (conflicting) with task2 than v7, i.e. averaging cancels more.
  (3) whole-Policy interference: ||0.5*(A+B)|| / (0.5*(||A||+||B||)) per tensor,
      lower = more destructive cancellation. Compare v7xv5 vs v9xv5.
"""
import os
import torch

R = r"C:\Users\shahvandit\Downloads\Drone23_curent\drone_battery1\results"
CK = {m: os.path.join(R, m, "Drone", "checkpoint.pt") for m in
      ("task1_v7", "task1_v9", "task2_v5")}

# obs layout (per memory / SimpleTask1 CollectObservations):
#  88 battery | 89,90 BS dir x,z | 95 BS dist | 96 charging_state
DIMS = {"battery(88)": 88, "BSdir_x(89)": 89, "BSdir_z(90)": 90,
        "BSdist(95)": 95, "charge(96)": 96, "PTdir_x(85?)": 85}


def policy(m):
    return torch.load(CK[m], map_location="cpu")["Policy"]


def find_input_proj(sd):
    for k in sd:
        if k.endswith("input_proj.weight"):
            return k, sd[k]              # [d_model, obs_dim]
    raise KeyError("no input_proj.weight")


def main():
    sds = {m: policy(m) for m in CK}
    k_ip, _ = find_input_proj(sds["task2_v5"])
    print(f"[input_proj key] {k_ip}  shape={tuple(sds['task2_v5'][k_ip].shape)}\n")

    W = {m: sds[m][k_ip] for m in CK}    # [128, 99]

    print("=== (1) input_proj column L2 norm per input dim ===")
    print(f"{'dim':<14}{'task1_v7':>10}{'task1_v9':>10}{'task2_v5':>10}")
    for name, j in DIMS.items():
        n7, n9, n5 = (W[m][:, j].norm().item() for m in ("task1_v7", "task1_v9", "task2_v5"))
        print(f"{name:<14}{n7:>10.4f}{n9:>10.4f}{n5:>10.4f}")

    print("\n=== (2) cosine(task1_col, task2_v5_col): >0 aligned, <0 conflicting ===")
    print(f"{'dim':<14}{'v7·v5':>10}{'v9·v5':>10}")
    cos = torch.nn.functional.cosine_similarity
    for name, j in DIMS.items():
        c7 = cos(W["task1_v7"][:, j], W["task2_v5"][:, j], dim=0).item()
        c9 = cos(W["task1_v9"][:, j], W["task2_v5"][:, j], dim=0).item()
        print(f"{name:<14}{c7:>10.3f}{c9:>10.3f}")

    print("\n=== (3) whole-Policy merge cancellation ratio at alpha=0.5 ===")
    print("    ratio = ||0.5(A+B)|| / (0.5(||A||+||B||));  1=no cancel, <1=interference")
    for tag, t1 in (("v7 x v5", "task1_v7"), ("v9 x v5", "task1_v9")):
        num = den = 0.0
        worst = []
        for key, b in sds["task2_v5"].items():
            a = sds[t1].get(key)
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                    and a.shape == b.shape and a.is_floating_point()):
                continue
            bl60 = (0.5 * (a + b)).norm().item()
            base = 0.5 * (a.norm().item() + b.norm().item())
            if base > 0:
                num += bl60; den += base
                worst.append((bl60 / base, key))
        print(f"  {tag}: global ratio = {num/den:.4f}")
        worst.sort()
        for r, key in worst[:5]:
            print(f"      {r:.3f}  {key}")


if __name__ == "__main__":
    main()
