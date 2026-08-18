from model import ManifoldModelFramework, ManifoldModelFrameworkV2
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
import numpy as np

from sphere_regression_test import build, Config, branch_mse, latent_alignment, Result, parse_args, aggregate, ARMS, mlp, FlatEuclidean
from manifold import *
from dataclasses import dataclass, field

import pdb
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.decomposition import PCA



def generate_swiss_roll_dataset(
    n_samples: int = 10000,
    noise: float = 0.0,
    seed: int = 42,
    device: str = "cuda"
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

    return x, x


class RandomObservationMap(nn.Module):
    """
    Fixed smooth nonlinear map f: z -> x.

    The weights are initialized once and then frozen. This is the
    observation function used to generate the synthetic dataset.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
    ):
        super().__init__()

        layers = []

        in_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # This is a data-generating function, not a trainable model.
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z):
        return self.net(z)
    

def generate_v2_dataset(
    n_samples: int = 10000,
    noise: float = 0.0,
    seed: int = 42,
    manifolds: list[Manifold] = None,
    x_dim: int = 32,
    hidden_dim: int = 64,
    f_depth: int = 2,
    device: str = "cuda",
):
    
    """Generates v2 dataset, that is a set of hidden variables z = (z1, z2, ...) where zi ~ Mi
    We then have some f we pass through such that f(z) = x, returns dataset with x, z pairs
    
    z_components is each of the points seperated by manifold
    
    :return: _description_
    :rtype: _type_
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(seed)

    z_components = []

    for manifold in manifolds:
        z_i = manifold.sample(n_samples)
        z_i = z_i.to(device)

        # Flatten the manifold's ambient representation
        z_i = z_i.reshape(n_samples, -1)

        z_components.append(z_i)

    # Product-space representation
    z = torch.cat(z_components, dim=-1)

    # Frozen observation function
    f = RandomObservationMap(
        input_dim=z.shape[-1],
        output_dim=x_dim,
        hidden_dim=hidden_dim,
        depth=f_depth,
    ).to(device)

    f.eval()

    with torch.no_grad():
        x = f(z)

        if noise > 0:
            x += noise * torch.randn_like(x)

    return x, z_components, z


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def buildv2(cfg: Config, manifolds: list[Manifold], encoder_in: int) -> ManifoldModelFrameworkV2:
    return ManifoldModelFrameworkV2(
        manifolds=manifolds,
        encoder=mlp(encoder_in, cfg.encoder_out, cfg.hidden, cfg.encoder_depth, cfg.residual),
        decoder=mlp(cfg.decoder_in, cfg.out_dim, cfg.hidden, cfg.decoder_depth, cfg.residual),
        encoder_out=cfg.encoder_out,
        decoder_in=cfg.decoder_in,
    ).to(cfg.device)

def run_onev1(cfg: Config, arm: str, seed: int, training_dataset: Dataset, test_dataset: Dataset, model: ManifoldModelFramework) -> Result:
    """Train each branch to solve the task alone, evaluate on the arg-max branch.

    The objective is ``sum_k gate_k * mse_k`` -- the gate weighting *losses*, not
    outputs -- so every manifold is trained to be individually competent and the
    arg-max read-out used at inference is asking the branches for something they
    were actually trained to do.  Both read-outs are still reported: with this
    objective they should nearly coincide, and any remaining gap means the gate
    is still spreading mass over branches of differing quality.
    """
    torch.manual_seed(seed)

    # V1
    model = build(cfg, arm)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    train_loader = DataLoader(
            training_dataset,
            batch_size=128,
            shuffle=True
    )
    test_loader = DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=True
        )
    
    train_loss_curve = []
    test_loss_curve = []
    for _ in range(cfg.steps):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x_train, y_train in train_loader:
            device = next(model.parameters()).device
            x_train, y_train = x_train.to(device), y_train.to(device)  # Move input tensor to GPU
            
            # V1 manifold
            outputs, logits = model(x_train)
            loss, _ = model.gated_loss(branch_mse(outputs, y_train), logits)
            
                        
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        
            train_loss += loss.item()
            train_samples += x_train.size(0)
            train_mse = train_loss / train_samples


        sched.step()
        train_loss /= train_samples
        train_loss_curve.append(train_loss)

        model.eval()
        test_loss = 0.0
        test_samples = 0

        with torch.no_grad():
            for x_test, y_test in test_loader:
                # Under the hard read-out, so train and test are directly comparable.
                x_test, y_test = x_test.to(device), y_test.to(device) 
                
                
                # V1 manifold
                outputs, logits = model(x_test)   # one pass, both read-outs

                soft, _ = model.combine(outputs, logits)
                hard, gate, _ = model.select(outputs, logits)

                test_mse = F.mse_loss(hard, y_test).item()
                test_mse_soft = F.mse_loss(soft, y_test).item()
                
                
                # V2 manifold
                # outputs = model(x_test)
                # test_mse = F.mse_loss(outputs, y_test)
                
                
                # Targets are standardized, so var(y) ~ 1 -- but compute it honestly.
                r2 = 1.0 - test_mse / y_test.var(unbiased=False).item()
        
                test_loss += loss.item()
                test_samples += x_test.size(0)

            test_loss /= test_samples
            test_loss_curve.append(test_loss)

    model.eval()
    final_predictions = []
    final_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:

            device = next(model.parameters()).device
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # V1
            outputs, logits = model(x_test)

            # hard argmax branch prediction
            predictions, _, _ = model.select(outputs, logits)

            final_predictions.append(predictions.cpu())
            final_targets.append(y_test.cpu())
            
            # V2
            # outputs = model(x_test)
            # test_mse = F.mse_loss(outputs, y_test)
            
    final_predictions = torch.cat(final_predictions, dim=0)
    final_targets = torch.cat(final_targets, dim=0)


    return Result(
        test_mse=test_mse,
        # V1
        test_mse_soft=test_mse_soft,
        # V2
        # test_mse_soft=0,
        train_mse=train_mse,
        r2=r2,
        alignment=None,
        # V1
        gate=gate.mean(0).tolist(),
        # V2
        # gate=None,
        params=sum(p.numel() for p in model.parameters()),
        train_loss_curve=train_loss_curve,
        test_loss_curve=test_loss_curve,
        targets=final_targets,
        predictions=final_predictions
    )


def run_onev2(cfg: Config, arm: str, seed: int, training_dataset: Dataset, test_dataset: Dataset, model: ManifoldModelFramework) -> Result:
    """Train each branch to solve the task alone, evaluate on the arg-max branch.

    The objective is ``sum_k gate_k * mse_k`` -- the gate weighting *losses*, not
    outputs -- so every manifold is trained to be individually competent and the
    arg-max read-out used at inference is asking the branches for something they
    were actually trained to do.  Both read-outs are still reported: with this
    objective they should nearly coincide, and any remaining gap means the gate
    is still spreading mass over branches of differing quality.
    """
    torch.manual_seed(seed)

    # V1
    # model = build(cfg, arm)
    
    # V2
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    train_loader = DataLoader(
            training_dataset,
            batch_size=128,
            shuffle=True
    )
    test_loader = DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=True
        )
    
    train_loss_curve = []
    test_loss_curve = []
    for _ in range(cfg.steps):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x_train, y_train in train_loader:
            device = next(model.parameters()).device
            x_train, y_train = x_train.to(device), y_train.to(device)  # Move input tensor to GPU
            
            # V1 manifold
            # outputs, logits = model(x_train)
            # loss, _ = model.gated_loss(branch_mse(outputs, y_train), logits)
            
            # V2 manifold
            outputs = model(x_train) 
            loss = F.mse_loss(outputs, y_train)                
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        
            train_loss += loss.item()
            train_samples += x_train.size(0)
            train_mse = train_loss / train_samples


        sched.step()
        train_loss /= train_samples
        train_loss_curve.append(train_loss)

        model.eval()
        test_loss = 0.0
        test_samples = 0

        with torch.no_grad():
            for x_test, y_test in test_loader:
                # Under the hard read-out, so train and test are directly comparable.
                x_test, y_test = x_test.to(device), y_test.to(device) 
                
                
                # V1 manifold
                # outputs, logits = model(x_test)   # one pass, both read-outs

                # soft, _ = model.combine(outputs, logits)
                # hard, gate, _ = model.select(outputs, logits)

                # test_mse = F.mse_loss(hard, y_test).item()
                # test_mse_soft = F.mse_loss(soft, y_test).item()
                
                
                # V2 manifold
                outputs = model(x_test)
                test_mse = F.mse_loss(outputs, y_test)
                
                
                # Targets are standardized, so var(y) ~ 1 -- but compute it honestly.
                r2 = 1.0 - test_mse / y_test.var(unbiased=False).item()
        
                test_loss += loss.item()
                test_samples += x_test.size(0)

            test_loss /= test_samples
            test_loss_curve.append(test_loss)

    model.eval()
    final_predictions = []
    final_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:

            device = next(model.parameters()).device
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # V1
            # outputs, logits = model(x_test)

            # # hard argmax branch prediction
            # predictions, _, _ = model.select(outputs, logits)

            # final_predictions.append(predictions.cpu())
            # final_targets.append(y_test.cpu())
            
            # V2
            outputs = model(x_test)
            test_mse = F.mse_loss(outputs, y_test)
            
    # final_predictions = torch.cat(final_predictions, dim=0)
    # final_targets = torch.cat(final_targets, dim=0)

    return Result(
        test_mse=test_mse,
        # V1
        # test_mse_soft=test_mse_soft,
        # V2
        test_mse_soft=0,
        train_mse=train_mse,
        r2=r2,
        alignment=None,
        # V1
        # gate=gate.mean(0).tolist(),
        # V2
        gate=None,
        params=sum(p.numel() for p in model.parameters()),
        train_loss_curve=train_loss_curve,
        test_loss_curve=test_loss_curve,
        targets=final_targets,
        predictions=final_predictions
    )


def product_distance(
    z: list[Tensor],
    z_prime: list[Tensor],
    manifolds: list[Manifold],
) -> Tensor:
    """
    Geodesic distance on a product manifold.

    z[i] and z_prime[i] belong to manifolds[i].

    Each tensor should have shape (..., ambient_dim).
    """

    if len(z) != len(z_prime) or len(z) != len(manifolds):
        raise ValueError("Mismatched number of manifold components.")

    squared_distances = []

    for manifold, zi, zi_prime in zip(
        manifolds,
        z,
        z_prime,
    ):
        d = manifold.distance(zi, zi_prime)
        squared_distances.append(d ** 2)

    return torch.sqrt(
        torch.stack(squared_distances, dim=-1).sum(dim=-1)
    )


def pairwise_product_distance(
    Z: list[Tensor],
    manifolds: list[Manifold],
) -> Tensor:
    """
    Compute all pairwise geodesic distances between
    points on a product manifold.

    Z[i]: [N, ambient_dim_i]

    Returns:
        [N, N] pairwise distance matrix.
    """

    n = Z[0].shape[0]

    D2 = torch.zeros(
        n,
        n,
        device=Z[0].device,
        dtype=Z[0].dtype,
    )

    for manifold, z in zip(manifolds, Z):

        # [N, 1, d]
        x = z.unsqueeze(1)

        # [1, N, d]
        y = z.unsqueeze(0)

        d = manifold.distance(x, y)

        D2 += d ** 2

    return torch.sqrt(D2)


def knn_preservation(
    D_true: Tensor,
    D_learned: Tensor,
    ks: list[int] = [5, 10, 20],
) -> dict[int, float]:
    """
    Measure KNN preservation between two pairwise distance matrices.

    Args:
        D_true: [N, N] ground-truth distance matrix.
        D_learned: [N, N] learned distance matrix.
        ks: Values of k to evaluate.

    Returns:
        Dictionary mapping k -> mean KNN overlap.
    """

    if D_true.shape != D_learned.shape:
        raise ValueError(
            f"Distance matrices must have the same shape, "
            f"got {D_true.shape} and {D_learned.shape}"
        )

    N = D_true.shape[0]

    # Make sure a point cannot be its own nearest neighbor.
    D_true = D_true.clone()
    D_learned = D_learned.clone()

    D_true.fill_diagonal_(float("inf"))
    D_learned.fill_diagonal_(float("inf"))

    results = {}

    for k in ks:
        if k >= N:
            raise ValueError(
                f"k={k} must be smaller than number of points N={N}"
            )

        # [N, k]
        true_neighbors = torch.topk(
            D_true,
            k=k,
            dim=1,
            largest=False,
        ).indices

        learned_neighbors = torch.topk(
            D_learned,
            k=k,
            dim=1,
            largest=False,
        ).indices

        # Compare each row's neighbor sets.
        #
        # [N, k, 1] == [N, 1, k]
        # gives [N, k, k], where each entry says whether
        # a true neighbor matches a learned neighbor.
        matches = (
            true_neighbors.unsqueeze(-1)
            == learned_neighbors.unsqueeze(-2)
        )

        # Number of unique true neighbors that also appear
        # in the learned neighborhood.
        overlap = matches.any(dim=-1).sum(dim=-1)

        # Normalize by k.
        scores = overlap.float() / k

        results[k] = scores.mean().item()

    return results


def visualize_model_outputs(
    model: ManifoldModelFrameworkV2,
    X: Tensor,
    z_components: list[Tensor],
    model_name: str,
    save_dir: str = "/scratch/wongbr55/LatentManifoldRegularizer",
    n_frames: int = 60,
    fps: int = 10,
    max_points: int = 5000,
):
    """
    Visualizes learned manifold representations together with
    their corresponding ground-truth manifold representations.

    z_components[i] corresponds to model.manifolds[i].

    For each manifold, the generated/learned points and ground-truth
    points are plotted together in the same rotating GIF.
    """

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # ---------------------------------------------------------
    # Compute learned manifold representations.
    # ---------------------------------------------------------

    with torch.no_grad():
        h = model.encoder(X)

        manifold_outputs = []

        for manifold, to_ambient in zip(
            model.manifolds,
            model.to_ambient,
        ):
            z = to_ambient(h)
            z = manifold.project(z)
            manifold_outputs.append(z)

    # ---------------------------------------------------------
    # Validate ground-truth components.
    # ---------------------------------------------------------

    if len(z_components) != len(model.manifolds):
        raise ValueError(
            f"Expected {len(model.manifolds)} z components, "
            f"got {len(z_components)}."
        )

    n = X.shape[0]

    for i, (manifold, z_true, z_learned) in enumerate(
        zip(
            model.manifolds,
            z_components,
            manifold_outputs,
        )
    ):

        if z_true.shape[0] != n:
            raise ValueError(
                f"z_components[{i}] has {z_true.shape[0]} points, "
                f"but X has {n} points."
            )

        if z_true.shape[-1] != manifold.ambient_dim:
            raise ValueError(
                f"z_components[{i}] has ambient dimension "
                f"{z_true.shape[-1]}, but {repr(manifold)} "
                f"expects {manifold.ambient_dim}."
            )

        if z_learned.shape[-1] != manifold.ambient_dim:
            raise ValueError(
                f"Learned output for {repr(manifold)} has ambient "
                f"dimension {z_learned.shape[-1]}, but expected "
                f"{manifold.ambient_dim}."
            )

    # ---------------------------------------------------------
    # Move everything to CPU for plotting.
    # ---------------------------------------------------------

    manifold_outputs = [
        z.detach().cpu()
        for z in manifold_outputs
    ]

    z_components_cpu = [
        z.detach().cpu()
        for z in z_components
    ]

    # ---------------------------------------------------------
    # Subsample.
    #
    # IMPORTANT:
    # Use exactly the same indices for both representations.
    # ---------------------------------------------------------

    if n > max_points:
        indices = np.random.choice(
            n,
            size=max_points,
            replace=False,
        )
    else:
        indices = np.arange(n)

    # ---------------------------------------------------------
    # Plot each manifold.
    # ---------------------------------------------------------

    for i, (manifold, learned, true) in enumerate(
        zip(
            model.manifolds,
            manifold_outputs,
            z_components_cpu,
        )
    ):

        learned_points = learned[indices]
        true_points = true[indices]

        ambient_dim = learned_points.shape[-1]

        # -----------------------------------------------------
        # Convert both representations to 3D.
        # -----------------------------------------------------

        if ambient_dim == 3:

            learned_points_3d = learned_points
            true_points_3d = true_points

        elif ambient_dim > 3 and isinstance(manifold, Torus):

            learned_points_3d = manifold.torus_to_3d(
                learned_points
            )

            true_points_3d = manifold.torus_to_3d(
                true_points
            )

        elif ambient_dim > 3:

            # Use PCA based on BOTH sets so that they are
            # represented in the same coordinate system.
            combined = np.concatenate(
                [true_points, learned_points],
                axis=0,
            )

            pca = PCA(n_components=3)
            pca.fit(combined)

            true_points_3d = pca.transform(true_points)
            learned_points_3d = pca.transform(learned_points)

        elif ambient_dim == 2:
            
            true_points_3d = np.zeros(
                (true_points.shape[0], 3),
                dtype=true_points.dtype,
            )

            learned_points_3d = np.zeros(
                (learned_points.shape[0], 3),
                dtype=learned_points.dtype,
            )

            true_points_3d[:, :2] = true_points
            learned_points_3d[:, :2] = learned_points

        else:
            raise ValueError(
                f"Cannot visualize ambient dimension "
                f"{ambient_dim}"
            )

        # -----------------------------------------------------
        # Create figure.
        # -----------------------------------------------------

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Ground truth
        ax.scatter(
            true_points_3d[:, 0],
            true_points_3d[:, 1],
            true_points_3d[:, 2],
            s=8,
            alpha=0.5,
            label="Ground truth",
        )

        # Learned
        ax.scatter(
            learned_points_3d[:, 0],
            learned_points_3d[:, 1],
            learned_points_3d[:, 2],
            s=8,
            alpha=0.5,
            label="Learned",
        )

        ax.set_title(
            f"{model_name}: {repr(manifold)}"
        )

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

        ax.legend()

        # -----------------------------------------------------
        # Equal aspect ratio.
        #
        # Calculate bounds from BOTH representations.
        # -----------------------------------------------------

        combined_points = np.concatenate(
            [
                true_points_3d,
                learned_points_3d,
            ],
            axis=0,
        )

        max_range = (
            combined_points.max(axis=0)
            - combined_points.min(axis=0)
        ).max() / 2.0

        if max_range == 0:
            max_range = 1.0

        mid = (
            combined_points.max(axis=0)
            + combined_points.min(axis=0)
        ) / 2.0

        ax.set_xlim(
            mid[0] - max_range,
            mid[0] + max_range,
        )

        ax.set_ylim(
            mid[1] - max_range,
            mid[1] + max_range,
        )

        ax.set_zlim(
            mid[2] - max_range,
            mid[2] + max_range,
        )

        # -----------------------------------------------------
        # Rotation.
        # -----------------------------------------------------

        def update(frame):
            ax.view_init(
                elev=30,
                azim=frame * 360 / n_frames,
            )
            return (ax,)

        animation = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 / fps,
            blit=False,
        )

        filename = os.path.join(
            save_dir,
            f"{i}_{model_name}_comparison.gif",
        )

        animation.save(
            filename,
            writer=PillowWriter(fps=fps),
        )

        plt.close(fig)

        print(f"Saved: {filename}")
    

def geodesic_preservation(save_dir, model_name, D_true, D_learned):
    mask = torch.triu(
        torch.ones_like(D_true, dtype=torch.bool),
        diagonal=1,
        )

    true_dist = D_true[mask]
    learned_dist = D_learned[mask]
        
    plt.figure(figsize=(7, 7))

    plt.scatter(
            true_dist.cpu(),
            learned_dist.cpu(),
            s=2,
            alpha=0.2,
        )

    max_dist = max(
            true_dist.max().item(),
            learned_dist.max().item(),
        )

    plt.plot(
            [0, max_dist],
            [0, max_dist],
            linestyle="--",
        )

    plt.xlabel("True geodesic distance")
    plt.ylabel("Learned geodesic distance")
    plt.title("Geodesic Distance Preservation")

    plt.tight_layout()
    plt.savefig(
            f"{save_dir}/geodesic_distance_preservation.png",
            dpi=300,
        )

    plt.close()
    
        # Pearson
    pearson = torch.corrcoef(
        torch.stack([
            true_dist,
            learned_dist,
        ])
    )[0, 1]

    # Stress
    stress = torch.sqrt(
        ((learned_dist - true_dist) ** 2).sum()
        / (true_dist ** 2).sum()
    )

    # Relative error
    relative_error = (
        (learned_dist - true_dist).abs()
        / (true_dist + 1e-8)
    )

    return {
            "pearson": pearson.item(),
            "stress": stress.item(),
            "mean_relative_error": relative_error.mean().item(),
            "median_relative_error": relative_error.median().item(),
        }


def learnt_latent_analysis(model: ManifoldModelFrameworkV2, X: Tensor, Z: Tensor, z_components: list, model_name: str, save_dir: str = "/scratch/wongbr55/LatentManifoldRegularizer", n_frames: int = 60,
    fps: int = 10,
    max_points: int = 5000,):
    """
    Performs V2 analysis, i.e. checks if interolation works well, if distances are preserved, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    visualize_model_outputs(model, X, z_components, model_name, save_dir, n_frames, fps, max_points)
    
    with torch.no_grad():
        D_true = pairwise_product_distance(
            z_components,
            manifolds,
        )
        
        z_learned_components = model.latent_representations(X)

        D_learned = pairwise_product_distance(
            z_learned_components,
            manifolds,
        )
        
        D_product = product_distance(z_components, z_learned_components, model.manifolds)
        
        knn_metrics = knn_preservation(
            D_true,
            D_learned,
            ks=[5, 10, 20, 50],
        )
        
        distance_metrics = geodesic_preservation(save_dir, model_name, D_true, D_learned)

        return {"knn_metrics" : knn_metrics, "distance_metrics" : distance_metrics}


def print_analysis_results(analysis_results, i):
    print("=" * 50)
    print(f"LATENT SPACE ANALYSIS FOR RUN {i}")
    print("=" * 50)

    # KNN metrics
    print("\nKNN Preservation")
    print("-" * 30)

    for k, score in analysis_results["knn_metrics"].items():
        print(f"KNN@{k:<3}: {score:.4f}")

    # Distance metrics
    print("\nDistance Preservation")
    print("-" * 30)

    distance_metrics = analysis_results["distance_metrics"]

    print(f"{'Pearson':<25}: {distance_metrics['pearson']:.4f}")
    print(f"{'Stress':<25}: {distance_metrics['stress']:.4f}")
    print(f"{'Mean relative error':<25}: {distance_metrics['mean_relative_error']:.4f}")
    print(f"{'Median relative error':<25}: {distance_metrics['median_relative_error']:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    
    # X, Y = generate_swiss_roll_dataset(
    # n_samples=1000,
    # noise=0.05
    # )
    
    manifolds = [Hyperbolic(3)]
    
    X, z_components, Z = generate_v2_dataset(
        n_samples=1000,
        noise=0,
        seed=0,
        manifolds=manifolds)

    cfg, arms = parse_args()
    # V2
    sum_dim = sum([m.dim for m in manifolds])
    cfg.out_dim = X.shape[1]
    euclid_manifold = [FlatEuclidean(sum_dim)]
    
    # model = buildv2(cfg, manifolds, X.shape[1])
    dataset = CustomDataset(X, X)
    
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2]) 
    
    
    d1, d2, d3, d4, d5 = random_split(train_dataset, [0.2 for __ in range(0, 5)])
    partition_train_datasets = [d1, d2, d3, d4, d5]
    
    final_train_mse = []
    final_test_mse = []
    
    final_base_train_mse = []
    final_base_test_mse = []
    for i in range(0, len(partition_train_datasets)):
        model = buildv2(cfg, manifolds, X.shape[1])
        baseline_model = buildv2(cfg, euclid_manifold, X.shape[1])
        
        data_so_far = combined_dataset = ConcatDataset(partition_train_datasets[:i + 1])
        
        results = run_onev2(cfg, arms[0], 0, data_so_far, test_dataset, model)
        baseline_results = run_onev2(cfg, arms[0], 0, data_so_far, test_dataset, baseline_model)
        
        analysis_results = learnt_latent_analysis(model, X, Z, z_components, f"latent_model_data_partition_{i}", "/scratch/wongbr55/LatentManifoldRegularizer/V2Analysis/Hyperbolic")
        print_analysis_results(analysis_results, i)
        final_train_mse.append(results.train_mse)
        final_test_mse.append(results.test_mse)
        final_base_train_mse.append(baseline_results.train_mse)
        final_base_test_mse.append(baseline_results.test_mse)
    
    print("\n" + "=" * 80)
    print(f"{'Run':<8} {'Train MSE':>15} {'Test MSE':>15} {'Base Train MSE':>18} {'Base Test MSE':>18}")
    print("-" * 80)

    for i, (train, test, base_train, base_test) in enumerate(
        zip(
            final_train_mse,
            final_test_mse,
            final_base_train_mse,
            final_base_test_mse
        )
    ):
        print(
            f"{i:<8} "
            f"{train:>15.6f} "
            f"{test:>15.6f} "
            f"{base_train:>18.6f} "
            f"{base_test:>18.6f}"
        )

    print("=" * 80)
    
    # results = run_one(cfg, arms[0], 0, train_dataset, test_dataset, model)
    # test_mse = results.test_mse
    # train_mse = results.train_mse
    # print(f"ran train MSE {train_mse:.5f}")
    # print(f"ran test MSE {test_mse:.5f}")
    
    # run analysis
    
    
    # V1
    rows: list[tuple[str, Result, tuple[float, float], tuple[float, float], float]] = []
    for arm in arms:
        results = [run_onev1(cfg, arm, seed, train_dataset, test_dataset, None) for seed in range(cfg.seeds)]
        mse = aggregate([r.test_mse for r in results])
        r2 = aggregate([r.r2 for r in results])
        soft = aggregate([r.test_mse_soft for r in results])[0]
        rows.append((arm, results[0], mse, r2, soft))
        print(f"  ran {arm:<13} test MSE {mse[0]:.5f} +- {mse[1]:.5f}")

    print(f"\n{'arm':<13} {'test MSE (hard)':>18} {'soft':>9} {'train MSE':>10} {'R^2':>8} "
            f"{'align':>8} {'params':>9}  description")
    print("-" * 114)
    for arm, first, mse, r2, soft in rows:
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
    
    plt.figure(figsize=(8, 6))

    n_plots = len(rows)

    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig = plt.figure(
        figsize=(9 * n_cols, 8 * n_rows)
    )

    for idx, (arm, first, _, _, _) in enumerate(rows):

        true = first.targets.numpy()
        pred = first.predictions.numpy()

        ax = fig.add_subplot(
            n_rows,
            n_cols,
            idx + 1,
            projection="3d"
        )

        # Plot ground truth Swiss roll
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            s=5,
            color="black",
            label="Ground truth"
        )

        # Plot reconstructed points
        ax.scatter(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            s=12,
            alpha=0.9,
            color="red",
            label="Reconstruction"
        )

        ax.set_title(
            arm,
            fontsize=16
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.legend()

        # Keep same viewing angle
        ax.view_init(
            elev=20,
            azim=-60
        )


    plt.tight_layout()

    plt.savefig(
        "/scratch/wongbr55/latent_man_reg/swiss_roll_reconstructions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()