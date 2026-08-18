"""Target manifolds for regularizing latent representations.

Each class describes a manifold embedded in some ambient Euclidean space and
exposes a ``project`` map

    project : R^ambient_dim -> M

that is *surjective onto (the interior of) M* and differentiable almost
everywhere, so it can be dropped straight into a network's forward pass.  It is
not an orthogonal projection in general -- the only contract is that the output
is guaranteed to lie on the manifold, which is what we need to force a latent
space to live on M.

All methods accept batched tensors of shape ``(..., ambient_dim)`` and preserve
the leading dimensions, dtype and device of their input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor

# Guard against dividing by the norm of a (near) zero vector.
EPS = {torch.float16: 1e-4, torch.float32: 1e-7, torch.float64: 1e-15}


def _eps(x: Tensor) -> float:
    return EPS.get(x.dtype, 1e-7)


class Manifold(ABC):
    """Base class for a manifold sitting inside R^ambient_dim.

    Args:
        dim: intrinsic dimension of the manifold.
    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Dimension of the Euclidean space the manifold is embedded in."""

    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """Map an ambient vector onto the manifold."""

    @abstractmethod
    def contains(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        """Boolean mask of shape ``x.shape[:-1]``: does each row lie on M?"""
    
    @abstractmethod
    def sample(
        self,
        n_samples: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample n random points from a chosen distribution on the manifold."""
        
    def geodesic(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Evaluate the geodesic from x to y at time t.

        Default implementation is the Euclidean geodesic, i.e.
        straight-line interpolation.

        Args:
            x: Starting points, shape (..., ambient_dim).
            y: Ending points, shape (..., ambient_dim).
            t: Interpolation parameter. Typically in [0, 1].
               Can be scalar or broadcastable to x/y.

        Returns:
            Points along the geodesic, shape broadcastable to (..., ambient_dim).
        """
        self._check_ambient(x)
        self._check_ambient(y)

        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, "
                f"got {tuple(x.shape)} and {tuple(y.shape)}"
            )

        return (1 - t) * x + t * y
    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Return the geodesic distance between x and y.

        Default implementation is the Euclidean geodesic distance, i.e.
        straightL2 norm.

        Args:
            x: Starting points, shape (..., ambient_dim).
            y: Ending points, shape (..., ambient_dim).

        Returns:
            Geodesic distance
        """
        
        self._check_ambient(x)
        self._check_ambient(y)

        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))


    def _check_ambient(self, x: Tensor) -> None:
        if x.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"{type(self).__name__} expects vectors of size "
                f"{self.ambient_dim}, got {tuple(x.shape)}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"


class Sphere(Manifold):
    """The sphere S^dim of radius ``radius``, embedded in R^(dim+1).

    ``project`` is radial rescaling, ``x -> radius * x / ||x||``.  The origin has
    no well-defined direction, so vectors shorter than ``eps`` are sent to the
    north pole instead of producing NaNs.
    """

    def __init__(self, dim: int, radius: float = 1.0) -> None:
        super().__init__(dim)
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.radius = float(radius)

    @property
    def ambient_dim(self) -> int:
        return self.dim + 1

    def project(self, x: Tensor) -> Tensor:
        self._check_ambient(x)
        eps = _eps(x)
        norm = x.norm(dim=-1, keepdim=True)

        # Fall back to the north pole e_0 wherever x is degenerate.
        pole = torch.zeros_like(x)
        pole[..., 0] = 1.0
        x = torch.where(norm > eps, x, pole)
        norm = torch.where(norm > eps, norm, torch.ones_like(norm))

        return self.radius * x / norm

    def contains(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        self._check_ambient(x)
        return (x.norm(dim=-1) - self.radius).abs() <= atol
    
    def sample(
        self,
        n_samples: int,
        device="cpu",
        dtype=torch.float32,
        ):
        x = torch.randn(
            n_samples,
            self.ambient_dim,
            device=device,
            dtype=dtype,
        )
        self._check_ambient(x)
        x = x / x.norm(dim=-1, keepdim=True)

        return self.radius * x
    
    def geodesic(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ) -> Tensor:
        self._check_ambient(x)
        self._check_ambient(y)

        cos_theta = (x * y).sum(dim=-1, keepdim=True) / self.radius**2
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        # Handle x ≈ y.
        close = sin_theta.abs() < _eps(x)

        a = torch.sin((1 - t) * theta) / sin_theta.clamp_min(_eps(x))
        b = torch.sin(t * theta) / sin_theta.clamp_min(_eps(x))

        result = a * x + b * y

        # For nearly identical points, straight interpolation is sufficient.
        linear = (1 - t) * x + t * y
        result = torch.where(close, linear, result)

        return result
    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        self._check_ambient(x)
        self._check_ambient(y)

        cos_theta = (x * y).sum(dim=-1) / (self.radius ** 2)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        return self.radius * torch.acos(cos_theta)

    def __repr__(self) -> str:
        return f"Sphere(dim={self.dim}, radius={self.radius})"


class Hyperbolic(Manifold):
    """Hyperbolic space H^dim in the Poincare ball model.

    Points live in the open ball of radius ``1 / sqrt(c)`` in R^dim, where ``c``
    is the (positive) magnitude of the curvature -- so here ambient and
    intrinsic dimension coincide.

    ``project`` is the exponential map at the origin,

        x -> tanh(sqrt(c) ||x||) * x / (sqrt(c) ||x||),

    a smooth diffeomorphism from R^dim onto the open ball.  Unlike clipping, it
    never saturates the gradient to exactly zero and maps the whole of Euclidean
    space into the ball, which is what makes it usable as a latent bottleneck.
    A final clamp keeps points strictly inside the boundary in float arithmetic.
    """

    def __init__(self, dim: int, curvature: float = 1.0) -> None:
        super().__init__(dim)
        if curvature <= 0:
            raise ValueError(f"curvature must be > 0, got {curvature}")
        self.curvature = float(curvature)
        self.sqrt_c = self.curvature**0.5
        self.max_norm = 1.0 / self.sqrt_c

    @property
    def ambient_dim(self) -> int:
        return self.dim

    def project(self, x: Tensor) -> Tensor:
        self._check_ambient(x)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
        scale = torch.tanh(self.sqrt_c * norm) / (self.sqrt_c * norm)
        return self.clamp_to_ball(scale * x)

    def clamp_to_ball(self, x: Tensor, margin: float = 1e-5) -> Tensor:
        """Push points that floating point pushed onto/past the boundary back in."""
        self._check_ambient(x)
        limit = self.max_norm * (1.0 - margin)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
        return torch.where(norm > limit, x / norm * limit, x)

    def contains(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        self._check_ambient(x)
        return x.norm(dim=-1) < self.max_norm + atol

    def sample(
        self,
        n_samples: int,
        device="cuda",
        dtype=torch.float32,
        scale=1.0,
    ):
        x = scale * torch.randn(
            n_samples,
            self.ambient_dim,
            device=device,
            dtype=dtype,
        )
        self._check_ambient(x)
        return self.project(x)
    
    
    def lambda_x(self, x: Tensor) -> Tensor:
        """Conformal factor of the Poincare ball metric."""
        x2 = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.curvature * x2).clamp_min(_eps(x))


    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map on the Poincare ball.

        Maps a tangent vector v at x onto the manifold.
        """
        self._check_ambient(x)
        self._check_ambient(v)

        eps = _eps(x)

        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)

        lam = self.lambda_x(x)

        sqrt_c = self.sqrt_c

        factor = torch.tanh(
            sqrt_c * lam * v_norm / 2.0
        ) / (sqrt_c * v_norm)

        result = self.mobius_add(x, factor * v)

        return self.clamp_to_ball(result)


    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map on the Poincare ball.

        Returns the tangent vector at x pointing toward y.
        """
        self._check_ambient(x)
        self._check_ambient(y)

        eps = _eps(x)

        diff = self.mobius_add(-x, y)

        diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(eps)

        lam = self.lambda_x(x)

        factor = (
            2.0
            / (self.sqrt_c * lam)
            * torch.atanh(
                self.sqrt_c * diff_norm
            )
            / diff_norm
        )

        return factor * diff
    
    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Möbius addition in the Poincare ball of curvature -c.
        """
        self._check_ambient(x)
        self._check_ambient(y)

        c = self.curvature

        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        numerator = (
            (1 + 2 * c * xy + c * y2) * x
            + (1 - c * x2) * y
        )

        denominator = (
            1
            + 2 * c * xy
            + c**2 * x2 * y2
        ).clamp_min(_eps(x))

        return numerator / denominator
    def geodesic(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Geodesic from x to y in the Poincare ball.

        gamma(t) = Exp_x(t Log_x(y))
        """
        v = self.log_map(x, y)

        return self.exp_map(x, t * v)

    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        self._check_ambient(x)
        self._check_ambient(y)

        diff = self.mobius_add(-x, y)

        norm = diff.norm(dim=-1)

        arg = (
            self.sqrt_c * norm
        ).clamp(
            min=0.0,
            max=1.0 - _eps(x),
        )

        return (
            2.0 / self.sqrt_c
        ) * torch.atanh(arg)

    def __repr__(self) -> str:
        return f"Hyperbolic(dim={self.dim}, curvature={self.curvature})"


class Torus(Manifold):
    """The flat torus T^dim = S^1 x ... x S^1, embedded in R^(2*dim).

    Coordinates are stored as ``dim`` consecutive ``(cos, sin)`` pairs scaled by
    the corresponding radius, rather than as angles in [0, 2*pi).  The angular
    representation has a seam where the coordinate jumps, which breaks gradient
    flow; the circle-product embedding is smooth everywhere.

    ``project`` normalizes each 2D block independently, sending degenerate blocks
    to angle 0.

    Args:
        dim: number of circles.
        radii: a single radius for every circle, or one radius per circle.
    """

    def __init__(self, dim: int, radii: float | Sequence[float] = 1.0) -> None:
        super().__init__(dim)
        if isinstance(radii, (int, float)):
            radii_t = torch.full((dim,), float(radii))
        else:
            radii_t = torch.as_tensor(list(radii), dtype=torch.float32)
            if radii_t.shape != (dim,):
                raise ValueError(
                    f"radii must be a scalar or have length {dim}, "
                    f"got {tuple(radii_t.shape)}"
                )
        if (radii_t <= 0).any():
            raise ValueError("all radii must be > 0")
        self.radii = radii_t

    @property
    def ambient_dim(self) -> int:
        return 2 * self.dim

    def project(self, x: Tensor) -> Tensor:
        self._check_ambient(x)
        eps = _eps(x)
        pairs = x.unflatten(-1, (self.dim, 2))
        norm = pairs.norm(dim=-1, keepdim=True)

        # Degenerate circles fall back to (1, 0), i.e. angle 0.
        base = torch.zeros_like(pairs)
        base[..., 0] = 1.0
        pairs = torch.where(norm > eps, pairs, base)
        norm = torch.where(norm > eps, norm, torch.ones_like(norm))

        unit = pairs / norm
        return (unit * self._radii_like(x).unsqueeze(-1)).flatten(-2)

    def to_angles(self, x: Tensor) -> Tensor:
        """Ambient coordinates -> angles in (-pi, pi], shape ``(..., dim)``."""
        self._check_ambient(x)
        pairs = x.unflatten(-1, (self.dim, 2))
        return torch.atan2(pairs[..., 1], pairs[..., 0])

    def from_angles(self, theta: Tensor) -> Tensor:
        """Angles of shape ``(..., dim)`` -> a point on the torus."""
        if theta.shape[-1] != self.dim:
            raise ValueError(
                f"expected {self.dim} angles, got {tuple(theta.shape)}"
            )
        radii = self._radii_like(theta)
        pairs = torch.stack([theta.cos(), theta.sin()], dim=-1)
        return (pairs * radii.unsqueeze(-1)).flatten(-2)

    def contains(self, x: Tensor, atol: float = 1e-5) -> Tensor:
        self._check_ambient(x)
        pairs = x.unflatten(-1, (self.dim, 2))
        err = (pairs.norm(dim=-1) - self._radii_like(x)).abs()
        return (err <= atol).all(dim=-1)
    

    def sample(
        self,
        n_samples: int,
        device="cpu",
        dtype=torch.float32,
    ):
        theta = 2 * torch.pi * torch.rand(
            n_samples,
            self.dim,
            device=device,
            dtype=dtype,
        )
        res = self.from_angles(theta)
        self._check_ambient(res)
        return res
    
    def geodesic(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ) -> Tensor:
        self._check_ambient(x)
        self._check_ambient(y)

        x_pairs = x.unflatten(-1, (self.dim, 2))
        y_pairs = y.unflatten(-1, (self.dim, 2))

        theta_x = torch.atan2(
            x_pairs[..., 1],
            x_pairs[..., 0],
        )

        theta_y = torch.atan2(
            y_pairs[..., 1],
            y_pairs[..., 0],
        )

        # Shortest angular displacement.
        delta = torch.atan2(
            torch.sin(theta_y - theta_x),
            torch.cos(theta_y - theta_x),
        )

        theta = theta_x + t * delta

        radii = self._radii_like(theta)

        pairs = torch.stack(
            [
                theta.cos(),
                theta.sin(),
            ],
            dim=-1,
        )

        return (pairs * radii.unsqueeze(-1)).flatten(-2)


    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        self._check_ambient(x)
        self._check_ambient(y)

        x_pairs = x.unflatten(-1, (self.dim, 2))
        y_pairs = y.unflatten(-1, (self.dim, 2))

        theta_x = torch.atan2(
            x_pairs[..., 1],
            x_pairs[..., 0],
        )

        theta_y = torch.atan2(
            y_pairs[..., 1],
            y_pairs[..., 0],
        )

        delta = torch.atan2(
            torch.sin(theta_y - theta_x),
            torch.cos(theta_y - theta_x),
        )

        radii = self._radii_like(x)

        return torch.sqrt(
            (radii**2 * delta**2).sum(dim=-1)
        )


    def torus_to_3d(self, points: torch.Tensor, R: float = 2.0, r: float = 1.0):
        """
        Convert a T² point from its R⁴ product-of-circles representation

            (cos θ₁, sin θ₁, cos θ₂, sin θ₂)

        into the standard R³ torus embedding.

        Args:
            points: Tensor of shape (..., 4)
            R: Major radius of the torus.
            r: Minor radius of the torus.

        Returns:
            Tensor of shape (..., 3).
        """
        if points.shape[-1] != 4:
            raise ValueError(
                f"Expected last dimension to be 4, got {points.shape[-1]}"
            )

        pairs = points.reshape(*points.shape[:-1], 2, 2)

        theta1 = torch.atan2(
            pairs[..., 0, 1],
            pairs[..., 0, 0],
        )

        theta2 = torch.atan2(
            pairs[..., 1, 1],
            pairs[..., 1, 0],
        )

        x = (R + r * torch.cos(theta1)) * torch.cos(theta2)
        y = (R + r * torch.cos(theta1)) * torch.sin(theta2)
        z = r * torch.sin(theta1)

        return torch.stack([x, y, z], dim=-1)

    def _radii_like(self, x: Tensor) -> Tensor:
        return self.radii.to(device=x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"Torus(dim={self.dim}, radii={self.radii.tolist()})"


# if __name__ == "__main__":
#     torch.manual_seed(0)

#     manifolds = [
#         Sphere(dim=3, radius=2.0),
#         Hyperbolic(dim=4, curvature=0.5),
#         Torus(dim=3, radii=[1.0, 2.0, 0.5]),
#     ]

#     for m in manifolds:
#         x = torch.randn(8, m.ambient_dim, dtype=torch.float64) * 10.0
#         x[0] = 0.0  # degenerate input
#         y = m.project(x)
#         assert m.contains(y).all(), m
#         assert m.contains(m.project(y)).all(), f"{m} not idempotent-safe"
#         print(f"{m}: R^{m.ambient_dim} -> {tuple(y.shape)} ok")

#     # Gradients flow through every projection.
#     for m in manifolds:
#         x = torch.randn(4, m.ambient_dim, requires_grad=True)
#         m.project(x).sum().backward()
#         assert x.grad is not None and torch.isfinite(x.grad).all(), m
#     print("gradients ok")
