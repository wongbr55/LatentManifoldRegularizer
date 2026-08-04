from model import ManifoldModelFramework
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sphere_regression_test import build, Config, branch_mse, latent_alignment, Result, parse_args, aggregate, ARMS
from dataclasses import dataclass, field


def generate_swiss_roll(
    n_samples: int = 10000,
    noise: float = 0.0,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Generate Swiss roll data.

    Input:
        x: (N,3) points on the Swiss roll in R^3

    Target:
        y: (N,2) intrinsic coordinates (u,v) on the flat rectangle

    """

    torch.manual_seed(seed)

    # Intrinsic coordinates on the flat rectangle
    # u controls position along the roll
    # v controls height
    u = torch.rand(n_samples, device=device)
    v = torch.rand(n_samples, device=device)

    # Scale u to create several turns of the spiral
    t = 1.5 * torch.pi * (1 + 2*u)

    # Swiss roll embedding R^2 -> R^3
    x = torch.stack(
        [
            t * torch.cos(t),
            21 * v,
            t * torch.sin(t),
        ],
        dim=1
    )

    # Add optional noise in ambient space
    if noise > 0:
        x += noise * torch.randn_like(x)

    # Ground truth intrinsic coordinates
    target = torch.stack(
        [
            u,
            v
        ],
        dim=1
    )

    return x, target

class SwissRollDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

X, Y = generate_swiss_roll(
    n_samples=10000,
    noise=0.05
)

def run_one(cfg: Config, arm: str, seed: int, data: Dataset) -> Result:
    """Train each branch to solve the task alone, evaluate on the arg-max branch.

    The objective is ``sum_k gate_k * mse_k`` -- the gate weighting *losses*, not
    outputs -- so every manifold is trained to be individually competent and the
    arg-max read-out used at inference is asking the branches for something they
    were actually trained to do.  Both read-outs are still reported: with this
    objective they should nearly coincide, and any remaining gap means the gate
    is still spreading mass over branches of differing quality.
    """
    torch.manual_seed(seed)

    model = build(cfg, arm)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    model.train()
    for _ in range(cfg.steps):
        outputs, logits = model(data.x_train)
        loss, _ = model.gated_loss(branch_mse(outputs, data.y_train), logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        # Under the hard read-out, so train and test are directly comparable.
        train_out, train_logits = model(data.x_train)
        train_mse = F.mse_loss(model.select(train_out, train_logits)[0], data.y_train).item()

        outputs, logits = model(data.x_test)   # one pass, both read-outs
        soft, _ = model.combine(outputs, logits)
        hard, gate, _ = model.select(outputs, logits)

        test_mse = F.mse_loss(hard, data.y_test).item()
        test_mse_soft = F.mse_loss(soft, data.y_test).item()
        # Targets are standardized, so var(y) ~ 1 -- but compute it honestly.
        r2 = 1.0 - test_mse / data.y_test.var(unbiased=False).item()

    return Result(
        test_mse=test_mse,
        test_mse_soft=test_mse_soft,
        train_mse=train_mse,
        r2=r2,
        alignment=latent_alignment(model, data.x_test, data.p_test),
        gate=gate.mean(0).tolist(),
        params=sum(p.numel() for p in model.parameters()),
    )



if __name__ == "__main__":

    cfg, arms = parse_args()
    
    dataset = SwissRollDataset(X, Y)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )
    
    rows: list[tuple[str, Result, tuple[float, float], tuple[float, float], float]] = []
    for arm in arms:
        results = [run_one(cfg, arm, seed, dataset) for seed in range(cfg.seeds)]
        mse = aggregate([r.test_mse for r in results])
        r2 = aggregate([r.r2 for r in results])
        soft = aggregate([r.test_mse_soft for r in results])[0]
        rows.append((arm, results[0], mse, r2, soft))
        print(f"  ran {arm:<13} test MSE {mse[0]:.5f} +- {mse[1]:.5f}")

    print(f"\n{'arm':<13} {'test MSE (hard)':>18} {'soft':>9} {'train MSE':>10} {'R^2':>8} "
            f"{'align':>8} {'params':>9}  description")
    print("-" * 114)
    for arm, first, mse, r2, soft in sorted(rows, key=lambda r: r[2][0]):
        desc, _ = ARMS[arm]
        align = f"{first.alignment:6.1f}d" if first.alignment is not None else "     --"
        print(f"{arm:<13} {mse[0]:>10.5f} +-{mse[1]:<6.5f} {soft:>9.5f} {first.train_mse:>10.5f} "
                f"{r2[0]:>8.4f} {align:>8} {first.params:>9,}  {desc}")

    print("\nhard = arg-max branch only (inference); soft = gate-weighted mixture.")
    print("       Identical for single-manifold arms.  The loss is sum_k gate_k * mse_k, so")
    print("       each branch is trained to stand alone and the two should nearly agree.")
    print("align = mean geodesic angle between the latent and the true sphere coordinate,")
    print("        after optimal orthogonal alignment (single 3-d bottlenecks only).")
    print("test MSE and R^2 are mean +- std over seeds; train MSE is from the first seed.")