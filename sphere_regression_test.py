"""Does a manifold-constrained latent help when the data really lives on a manifold?

Task
----
Inputs are points on the unit sphere S^2 (coordinates in R^3).  Targets are one
random vector per sampled point -- but the vectors are *not* drawn
independently.  They are drawn jointly from a Gaussian process on the sphere, so
the values at nearby points are correlated and the whole assignment is a sample
path of a smooth random function ``f : S^2 -> R^out_dim``.  Nothing is expanded
in a basis: the field is defined only by its values at the points we sampled.

The covariance is the Gaussian (squared-exponential) kernel in chordal distance,

    k(p, q) = exp(-||p - q||^2 / (2 l^2)) = exp(-(1 - <p, q>) / l^2),

which is positive definite on S^2 because it is a Gaussian kernel of R^3
restricted to the sphere.  ``length_scale`` (l) is the difficulty knob and it
spans the whole range on its own: large l gives a field that varies slowly over
the sphere, small l decorrelates neighbouring points until the assignment really
is white noise and no model can generalize.  Marginal variance is 1 by
construction, so predicting the mean scores MSE = 1 and R^2 = 0.

This is the cleanest possible probe of the hypothesis behind
``ManifoldModelFramework``: the data manifold is *exactly* S^2, so a latent
bottleneck constrained to S^2 has precisely the right shape, while an
unconstrained R^3 bottleneck has to discover that shape from scratch.

Fairness
--------
Every arm -- including the unconstrained baselines -- runs through the *same*
``ManifoldModelFramework``.  The baseline uses ``FlatEuclidean``, a Manifold
whose ``project`` is the identity.  So the arms have identical parameter counts,
identical initialisation scheme and identical code path; the only difference is
whether ``project`` constrains the latent.  Comparing against a hand-written MLP
instead would confound the constraint with architectural differences.

Read-out
--------
Arms with several manifolds are trained on ``sum_k gate_k * mse_k`` -- the gate
weighting per-branch losses rather than being folded into one loss over the
mixed output -- so each manifold learns to solve the task by itself and the gate
learns which one fits.  Inference then commits to the single most probable
branch (``select``).  Weighting the *outputs* instead lets the branches split the
work into complementary halves that are each useless alone, which the arg-max
read-out would then expose as catastrophic error.

Train and test points are drawn from one joint field sample, so both splits are
values of the same function.  Each arm is run over several seeds and reported as
mean +- std; the dataset is built once per seed and shared by every arm.

    python sphere_regression_test.py --quick
    python sphere_regression_test.py --arms flat-3 sphere-2
    python sphere_regression_test.py --length-scale 0.25   # rougher field
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from manifold import Hyperbolic, Manifold, Sphere, Torus
from model import ManifoldModelFramework


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    # task
    out_dim: int = 8
    length_scale: float = 0.25  # GP correlation length in chordal distance on S^2
    # Sampling the field costs one Cholesky of an (train + test) square matrix,
    # so these are an order of magnitude smaller than a basis-expansion target
    # would allow.
    train_samples: int = 2048
    test_samples: int = 2048
    target_noise: float = 0.0  # observation noise added to the training targets
    input_noise: float = 0.0   # pushes inputs off the sphere (0 = exactly on S^2)

    # model.  Every hidden layer has width ``hidden``; the depths count hidden
    # layers, so the defaults (2, 1) reproduce the original two-block encoder and
    # one-block decoder exactly.
    hidden: int = 128
    encoder_out: int = 128
    decoder_in: int = 128
    encoder_depth: int = 2
    decoder_depth: int = 2
    residual: bool = False     # skip connections; only possible at uniform width

    # optimisation
    steps: int = 3000
    lr: float = 3e-3
    weight_decay: float = 0.0
    seeds: int = 3

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class FlatEuclidean(Manifold):
    """R^dim itself, with the identity as its projection.

    Lets the unconstrained baseline reuse ``ManifoldModelFramework`` verbatim, so
    the only thing that differs between arms is whether ``project`` does
    anything.  Everything lies on R^dim, hence ``contains`` is always True.
    """

    @property
    def ambient_dim(self) -> int:
        return self.dim

    def project(self, x: Tensor) -> Tensor:
        self._check_ambient(x)
        return x

    def contains(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        self._check_ambient(x)
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    def sample(self, n):
        return torch.randn(n, self.dim)

    def __repr__(self) -> str:
        return f"FlatEuclidean(dim={self.dim})"


# --------------------------------------------------------------------------- #
# Task: one random vector per point, correlated so the field is smooth
# --------------------------------------------------------------------------- #


def sample_sphere(n: int, generator: torch.Generator, device: str) -> Tensor:
    """Uniform points on S^2 (Gaussian directions, normalized)."""
    v = torch.randn(n, 3, generator=generator, device=device)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sphere_kernel(p: Tensor, length_scale: float) -> Tensor:
    """Gaussian kernel of the chordal distance between points on S^2.

    ``exp(-||p - q||^2 / (2 l^2))`` written via ``||p - q||^2 = 2 - 2 <p, q>``,
    so the diagonal is exactly 1 and no square roots are taken.  Positive
    definite because it is a Gaussian kernel of the ambient R^3.
    """
    cos = (p @ p.t()).clamp(-1.0, 1.0)
    return torch.exp((cos - 1.0) / length_scale**2)


def _matrix_sqrt_factor(k: Tensor) -> Tensor:
    """A factor ``L`` with ``L L^T ~ k``, for a smooth kernel matrix.

    A squared-exponential kernel matrix is badly rank-deficient in floating
    point -- its eigenvalues decay geometrically -- so a bare Cholesky fails.  We
    escalate a jitter (a nugget: it adds an i.i.d. component of variance
    ``jitter`` to the field, which is why we start small) and fall back to a
    clamped eigendecomposition, which cannot fail.
    """
    n = k.shape[0]
    eye = torch.eye(n, dtype=k.dtype, device=k.device)
    for exponent in range(-10, -2):
        factored = torch.linalg.cholesky_ex(k + (10.0**exponent) * eye)
        if int(factored.info) == 0:
            return factored.L

    values, vectors = torch.linalg.eigh(k)
    return vectors * values.clamp_min(0.0).sqrt()


def sample_field(
    p: Tensor, out_dim: int, length_scale: float, generator: torch.Generator
) -> Tensor:
    """Assign a vector in R^out_dim to every point in ``p``, jointly and smoothly.

    Draws ``out_dim`` independent sample paths of a mean-zero GP with covariance
    ``sphere_kernel`` at exactly the points given, then standardizes each output
    coordinate over the sample so a mean predictor scores MSE = 1.

    Args:
        p: ``(N, 3)`` points on the unit sphere.
        out_dim: width of the target vectors.
        length_scale: GP correlation length; smaller is rougher.
        generator: RNG for the draw.

    Returns:
        ``(N, out_dim)`` targets, in ``p``'s dtype and on its device.
    """
    # float64 throughout: the factorisation of a smooth kernel matrix is far too
    # ill-conditioned to trust in single precision.
    k = sphere_kernel(p.double(), length_scale)
    lower = _matrix_sqrt_factor(k)

    noise = torch.randn(p.shape[0], out_dim, generator=generator, device=p.device)
    y = lower @ noise.double()

    y = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True).clamp_min(1e-12)
    return y.to(p.dtype)


@dataclass
class Dataset:
    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    p_test: Tensor  # the true sphere coordinate behind each test input


def make_dataset(cfg: Config, generator: torch.Generator) -> Dataset:
    """One field sample, split into train and test.

    Both splits come from a single joint draw, so they are values of the *same*
    function -- which is what makes generalization measurable at all.  Targets
    are read off the clean sphere point even when the inputs are pushed off it.
    """
    n_train, n_test = cfg.train_samples, cfg.test_samples
    p = sample_sphere(n_train + n_test, generator, cfg.device)
    y = sample_field(p, cfg.out_dim, cfg.length_scale, generator)

    x = p
    if cfg.input_noise > 0:
        # Perturb off the sphere: the target still depends only on the direction,
        # so this tests whether a spherical bottleneck denoises the input.
        x = p + cfg.input_noise * torch.randn(p.shape, generator=generator, device=cfg.device)

    y_train = y[:n_train]
    if cfg.target_noise > 0:
        y_train = y_train + cfg.target_noise * torch.randn(
            y_train.shape, generator=generator, device=cfg.device
        )

    return Dataset(
        x_train=x[:n_train],
        y_train=y_train,
        x_test=x[n_train:],
        y_test=y[n_train:],
        p_test=p[n_train:],
    )


@torch.no_grad()
def field_diagnostics(data: Dataset, cfg: Config) -> tuple[float, float]:
    """(mean nearest-neighbour angle in degrees, kernel correlation at that angle).

    The correlation between a test point and its closest training point is the
    honest measure of how learnable the draw is: near 1 the field is smooth
    relative to the sampling density, near 0 the targets are effectively noise.
    """
    sim = (data.p_test @ F.normalize(data.x_train, dim=-1).t()).clamp(-1.0, 1.0)
    nearest = sim.max(dim=-1).values
    angle = math.degrees(nearest.arccos().mean().item())
    corr = math.exp((nearest.mean().item() - 1.0) / cfg.length_scale**2)
    return angle, corr


# --------------------------------------------------------------------------- #
# Model arms
# --------------------------------------------------------------------------- #


class ResidualBlock(nn.Module):
    """``x + W GELU(x)``, a pre-activation residual block of constant width.

    Only expressible because every hidden layer shares one width.  Without skips,
    a plain stack stops training well past a handful of layers, so a depth sweep
    on plain stacks alone cannot distinguish "depth does not help this task" from
    "this depth did not optimise" -- which is what these blocks are here to
    separate.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear(F.gelu(x))


def mlp(
    in_dim: int, out_dim: int, hidden: int, depth: int, residual: bool = False
) -> nn.Sequential:
    """``in_dim -> [hidden] * depth -> out_dim``, one uniform hidden width.

    Args:
        in_dim: input width.
        out_dim: output width.
        hidden: width of every hidden layer.
        depth: number of hidden layers.  ``0`` gives a bare linear map, and
            because nothing separates consecutive linear layers in that case,
            depth is the only thing that adds expressive power here -- stacked
            linear layers with no activation between them collapse to one.
        residual: insert skip connections around the interior blocks.

    Returns:
        The stack, as a ``Sequential``.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if depth == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
    for _ in range(depth - 1):
        if residual:
            layers.append(ResidualBlock(hidden))
        else:
            layers += [nn.GELU(), nn.Linear(hidden, hidden)]
    layers += [nn.GELU(), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


ARMS: dict[str, tuple[str, list[Manifold]]] = {
    # name              description                              manifolds
    "flat-3":       ("unconstrained R^3 bottleneck",             [FlatEuclidean(3)]),
    "flat-2":       ("unconstrained R^3 bottleneck",             [FlatEuclidean(2)]),
    "sphere-1":     ("S^1 in R^2",                               [Sphere(1)]),
    "sphere-2":     ("S^2 in R^3",                               [Sphere(2)]),
    "hyperbolic-3": ("Poincare ball H^3",                        [Hyperbolic(3)]),
    "hyperbolic-2": ("Poincare ball H^2",                        [Hyperbolic(2)]),
    "torus-2":      ("flat torus T^2 in R^4 ",                   [Torus(2)]),
    "torus-1":      ("flat torus T^2 in R^4 ",                   [Torus(1)]),
    "flat-16":      ("unconstrained R^16, wide bottleneck",      [FlatEuclidean(16)]),
    "sphere-15":    ("S^15 in R^16, wide bottleneck",            [Sphere(15)]),
    "flat-3x3":     ("3 unconstrained branches",                 [FlatEuclidean(3)] * 3),
    "mixture-3":    ("gated S^3 + H^3 + T^3, all intrinsic 3",   [Sphere(3), Hyperbolic(3), Torus(3)]),
}

DEFAULT_ARMS = ["flat-2", "sphere-1", "sphere-2", "hyperbolic-2", "torus-2"]


def build(cfg: Config, arm: str) -> ManifoldModelFramework:
    _, manifolds = ARMS[arm]
    return ManifoldModelFramework(
        manifolds=manifolds,
        encoder=mlp(3, cfg.encoder_out, cfg.hidden, cfg.encoder_depth, cfg.residual),
        decoder=mlp(cfg.decoder_in, cfg.out_dim, cfg.hidden, cfg.decoder_depth, cfg.residual),
        encoder_out=cfg.encoder_out,
        decoder_in=cfg.decoder_in,
    ).to(cfg.device)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@torch.no_grad()
def latent_alignment(model: ManifoldModelFramework, x: Tensor, p: Tensor) -> float | None:
    """How close is the learned latent to the true sphere coordinate?

    Only meaningful for a single 3-dimensional bottleneck.  We allow any
    orthogonal map (Procrustes, reflections included) because recovering the
    manifold up to an isometry is all we can ask for, then report the mean
    geodesic angle in degrees between the aligned latent and the true point.
    ``None`` when the arm's bottleneck isn't a 3-d sphere-shaped one.
    """
    if len(model.manifolds) != 1 or model.manifolds[0].ambient_dim != 3:
        return None

    h = model.encoder(x)
    z = model.manifolds[0].project(model.to_ambient[0](h))
    z = F.normalize(z, dim=-1)  # compare directions; flat arms aren't unit-norm

    # Optimal orthogonal R minimising ||z R - p|| is U V^T from svd(z^T p).
    u, _, vt = torch.linalg.svd(z.t() @ p)
    aligned = z @ (u @ vt)

    cos = (aligned * p).sum(-1).clamp(-1.0, 1.0)
    return math.degrees(cos.arccos().mean().item())


# --------------------------------------------------------------------------- #
# Train / evaluate one arm
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    test_mse: float       # hard read-out: arg-max branch only (the inference number)
    test_mse_soft: float  # soft read-out: gate-weighted mixture, for comparison
    train_mse: float      # hard read-out on the training split
    r2: float
    alignment: float | None
    gate: list[float]
    params: int
    
    train_loss_curve: list[float]
    test_loss_curve: list[float]
    predictions: list
    targets: list


def branch_mse(outputs: Tensor, y: Tensor) -> Tensor:
    """Per-sample MSE of every branch: ``(K, N, out)`` and ``(N, out)`` -> ``(K, N)``.

    Kept per-sample, rather than reduced to one scalar per branch, so the gate
    can weight branches differently for different points.
    """
    return (outputs - y).pow(2).mean(dim=-1)


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


def aggregate(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values)
    return t.mean().item(), t.std(unbiased=len(values) > 1).item()


def main(cfg: Config, arms: list[str]) -> None:
    print(f"device={cfg.device}  seeds={cfg.seeds}  steps={cfg.steps}")
    print(f"task: S^2 -> R^{cfg.out_dim}, GP field with chordal length scale "
          f"{cfg.length_scale}")
    print(f"data: {cfg.train_samples} train / {cfg.test_samples} test  "
          f"target_noise={cfg.target_noise}  input_noise={cfg.input_noise}")

    # One field sample per seed, shared by every arm: the arms must see identical
    # data, and the draw costs a Cholesky we do not want to repeat per arm.
    print("\nsampling fields", end="", flush=True)
    datasets = []
    for seed in range(cfg.seeds):
        gen = torch.Generator(device=cfg.device).manual_seed(seed)
        datasets.append(make_dataset(cfg, gen))
        print(".", end="", flush=True)
    angle, corr = field_diagnostics(datasets[0], cfg)
    print(f"\n  nearest training point is {angle:.1f}d away on average, "
          f"kernel correlation {corr:.3f}\n")

    rows: list[tuple[str, Result, tuple[float, float], tuple[float, float], float]] = []
    for arm in arms:
        results = [run_one(cfg, arm, seed, datasets[seed]) for seed in range(cfg.seeds)]
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

    gated = [(a, r) for a, r, _, _, _ in rows if len(ARMS[a][1]) > 1]
    if gated:
        print("\ngate usage (first seed)")
        for arm, first in gated:
            names = [type(m).__name__ for m in ARMS[arm][1]]
            usage = "  ".join(f"{n}={g:.2f}" for n, g in zip(names, first.gate))
            print(f"  {arm:<13} {usage}")


def parse_args() -> tuple[Config, list[str]]:
    cfg = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS),
                   help="which bottlenecks to compare")
    p.add_argument("--steps", type=int, default=cfg.steps)
    p.add_argument("--seeds", type=int, default=cfg.seeds)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--length-scale", type=float, default=cfg.length_scale,
                   help="GP correlation length; smaller = rougher field")
    p.add_argument("--target-noise", type=float, default=cfg.target_noise)
    p.add_argument("--input-noise", type=float, default=cfg.input_noise)
    p.add_argument("--train-samples", type=int, default=cfg.train_samples)
    p.add_argument("--test-samples", type=int, default=cfg.test_samples)
    p.add_argument("--hidden", type=int, default=cfg.hidden)
    p.add_argument("--encoder-depth", type=int, default=cfg.encoder_depth,
                   help="number of hidden layers in the encoder")
    p.add_argument("--decoder-depth", type=int, default=cfg.decoder_depth,
                   help="number of hidden layers in the decoder")
    p.add_argument("--residual", action="store_true",
                   help="skip connections around the interior blocks")
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--quick", action="store_true", help="fewer steps and one seed")
    a = p.parse_args()

    cfg = Config(
        steps=a.steps, seeds=a.seeds, lr=a.lr, length_scale=a.length_scale,
        target_noise=a.target_noise, input_noise=a.input_noise,
        train_samples=a.train_samples, test_samples=a.test_samples,
        hidden=a.hidden, encoder_depth=a.encoder_depth,
        decoder_depth=a.decoder_depth, residual=a.residual, device=a.device,
    )
    if a.quick:
        cfg.steps, cfg.seeds = 300, 1
        cfg.train_samples = cfg.test_samples = 512
    return cfg, a.arms


if __name__ == "__main__":
    main(*parse_args())
