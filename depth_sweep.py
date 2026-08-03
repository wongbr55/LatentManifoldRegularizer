"""Depth sweep: does a deeper encoder/decoder change the arm ordering?

Varies the number of hidden layers in *both* the encoder and decoder together,
with and without residual blocks (identical parameter counts either way), and
reuses one dataset per seed so every configuration sees the same targets.
"""

import time

import torch

from sphere_regression_test import Config, Dataset, make_dataset, run_one

DEPTHS = [2, 3]
ARMS = ["flat-3", "sphere-2", "flat-3x3", "mixture-3"]
SEEDS = 2
STEPS = 3000


def main() -> None:
    base = Config(steps=STEPS, seeds=SEEDS)
    print(f"device={base.device} steps={STEPS} seeds={SEEDS} "
          f"length_scale={base.length_scale} samples={base.train_samples}")

    # The dataset depends only on the task fields and the seed, none of which the
    # sweep varies -- so draw it once per seed rather than once per configuration.
    data: list[Dataset] = []
    for seed in range(SEEDS):
        gen = torch.Generator(device=base.device).manual_seed(seed)
        data.append(make_dataset(base, gen))

    rows = []
    started = time.time()
    for residual in (False, True):
        for depth in DEPTHS:
            cfg = Config(steps=STEPS, seeds=SEEDS, encoder_depth=depth,
                         decoder_depth=depth, residual=residual)
            for arm in ARMS:
                results = [run_one(cfg, arm, s, data[s]) for s in range(SEEDS)]
                mse = torch.tensor([r.test_mse for r in results])
                train = torch.tensor([r.train_mse for r in results])
                aligns = [r.alignment for r in results if r.alignment is not None]
                rows.append({
                    "residual": residual, "depth": depth, "arm": arm,
                    "mse": mse.mean().item(), "std": mse.std(unbiased=False).item(),
                    "train": train.mean().item(),
                    "align": sum(aligns) / len(aligns) if aligns else None,
                    "params": results[0].params,
                })
                r = rows[-1]
                print(f"  [{time.time()-started:6.0f}s] res={int(residual)} d={depth} "
                      f"{arm:<10} mse={r['mse']:.4f}")

    print(f"\n{'block':<10} {'depth':>5} {'arm':<10} {'test MSE':>16} "
          f"{'train MSE':>10} {'align':>8} {'params':>9}")
    print("-" * 74)
    for r in rows:
        block = "residual" if r["residual"] else "plain"
        align = f"{r['align']:6.1f}d" if r["align"] is not None else "     --"
        print(f"{block:<10} {r['depth']:>5} {r['arm']:<10} "
              f"{r['mse']:>8.4f} +-{r['std']:<5.4f} {r['train']:>10.4f} "
              f"{align:>8} {r['params']:>9,}")

    # The question is whether depth changes the *ordering*, not just the level.
    print("\nflat-3 minus sphere-2 (positive = sphere wins):")
    for residual in (False, True):
        block = "residual" if r["residual"] else "plain"
        line = []
        for depth in DEPTHS:
            got = {r["arm"]: r["mse"] for r in rows
                   if r["residual"] == residual and r["depth"] == depth}
            line.append(f"d={depth}: {got['flat-3'] - got['sphere-2']:+.4f}")
        print(f"  {'residual' if residual else 'plain':<9} " + "   ".join(line))


if __name__ == "__main__":
    main()
