from manifold import Manifold
import torch
from torch import Tensor
from torch import nn


class ManifoldModelFramework(nn.Module):
    
    """Simple implementation of neural network with induced manifold representation in hidden/latent space
    """
    
    def __init__(self, manifolds: list[Manifold], encoder: nn.Module, decoder: nn.Module, encoder_out: int, decoder_in: int):
        super().__init__()
        if len(manifolds) == 0:
            raise ValueError(f"Number of target manifolds must be >= 1, got 0")
        if encoder is None or decoder is None:
            raise ValueError(f"Encoder and decoder must not be None")
        if encoder_out < 1 or decoder_in < 1:
            raise ValueError(
                f"encoder_out and decoder_in must be >= 1, got {encoder_out} and {decoder_in}"
            )
        self.manifolds = manifolds
        self.num_manifolds = len(manifolds)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out = encoder_out
        self.decoder_in = decoder_in

        self.prob_manifold = nn.Linear(self.encoder_out, self.num_manifolds)

        # One head per manifold, mapping the encoder output into that manifold's
        # ambient space so ``manifold.project`` can be applied to it, and one head
        # mapping the projected point back to the decoder's input width.  The
        # ambient dimensions differ per manifold (dim+1 for a sphere, 2*dim for a
        # torus, ...), which is why these cannot be a single shared Linear.
        self.to_ambient = nn.ModuleList(
            nn.Linear(self.encoder_out, m.ambient_dim) for m in manifolds
        )
        self.from_ambient = nn.ModuleList(
            nn.Linear(m.ambient_dim, self.decoder_in) for m in manifolds
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the input through every manifold branch.

        Returns:
            outputs: decoder output of each branch, stacked on a leading axis of
                size ``num_manifolds``.  ``outputs[i]`` is the reconstruction
                obtained by forcing the latent onto ``self.manifolds[i]``.
            manifold_logits: *unnormalized* gating logits of shape
                ``(..., num_manifolds)``.  Apply ``softmax``/``log_softmax``
                downstream; keeping them raw here lets the caller use the
                numerically stable losses.
        """
        x = self.encoder(x)

        manifold_logits = self.prob_manifold(x)

        outputs = []
        for manifold, to_ambient, from_ambient in zip(
            self.manifolds, self.to_ambient, self.from_ambient
        ):
            # Into the manifold's ambient space, onto the manifold, then back out
            # to the width the decoder expects.
            z = to_ambient(x)
            z = manifold.project(z)
            z = from_ambient(z)
            outputs.append(self.decoder(z))

        return torch.stack(outputs), manifold_logits

    # ------------------------------------------------------------------ #
    # Read-outs
    #
    # ``forward`` deliberately returns every branch plus raw logits and takes no
    # stand on how to collapse them.  These two methods spell out the only thing
    # the logits are ever meant to be -- a categorical distribution over
    # branches -- so callers do not each re-derive the indexing.  Both take
    # ``forward``'s output rather than ``x``, so a single forward pass can serve
    # both read-outs.
    # ------------------------------------------------------------------ #

    @staticmethod
    def combine(outputs: Tensor, manifold_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Soft read-out: the gate-weighted convex combination of every branch.

        Differentiable in both the branch outputs and the gate, which is what
        makes it the read-out to *train* through: gradient reaches every
        manifold, weighted by how much the gate currently favours it, so the
        gate can learn which geometry fits without any branch being cut off.

        Args:
            outputs: ``(K, ..., P)`` branch outputs, as returned by ``forward``.
            manifold_logits: ``(..., K)`` raw gating logits.

        Returns:
            pred: ``(..., P)`` mixture ``sum_k gate_k * outputs_k``.
            gate: ``(..., K)`` gating probabilities.
        """
        gate = manifold_logits.softmax(dim=-1)
        weights = gate.movedim(-1, 0).unsqueeze(-1)   # (K, ..., 1)
        return (weights * outputs).sum(dim=0), gate

    @staticmethod
    def select(outputs: Tensor, manifold_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Hard read-out: the single most probable branch, per sample.

        The read-out for *inference*, when the question is "which manifold does
        this sample live on" and the answer should be one manifold rather than a
        blend of all of them.  Not differentiable through the choice (``argmax``
        has no gradient), so do not train on this directly without a
        straight-through or Gumbel estimator.

        Args:
            outputs: ``(K, ..., P)`` branch outputs, as returned by ``forward``.
            manifold_logits: ``(..., K)`` raw gating logits.

        Returns:
            pred: ``(..., P)`` output of the arg-max branch.
            gate: ``(..., K)`` gating probabilities.
            choice: ``(...)`` index of the chosen branch, in ``[0, K)``.
        """
        gate = manifold_logits.softmax(dim=-1)
        choice = gate.argmax(dim=-1)                  # (...)

        # Gather along the branch axis: index must match outputs' rank.
        index = choice.unsqueeze(0).unsqueeze(-1).expand(1, *choice.shape, outputs.shape[-1])
        pred = outputs.gather(0, index).squeeze(0)
        return pred, gate, choice

    @staticmethod
    def gated_loss(
        per_branch_losses: Tensor, manifold_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Gate-weighted mean of *per-branch* losses: ``sum_k gate_k * loss_k``.

        The counterpart to ``combine`` on the loss side, and the objective that
        makes ``select`` coherent.  Weighting the losses trains every branch to
        solve the task *on its own* -- scaled by how much the gate favours it --
        whereas putting the mixture inside a single loss
        (``loss(sum_k gate_k * out_k, y)``) only requires the blend to be right
        and lets the branches specialise into complementary pieces that are each
        useless alone.  Since inference commits to one branch, the branches have
        to be individually competent, so the loss must be the outer sum.

        The gate is differentiable here, and its gradient moves mass toward
        whichever branches have the lowest loss.  Left alone that drives the gate
        to collapse onto a single branch -- desirable when the point is to *pick*
        a manifold, but add an entropy bonus on ``gate`` if a diffuse gate is
        wanted instead.

        Args:
            per_branch_losses: ``(K, ...)`` per-sample loss for each branch, with
                the branch axis leading, matching ``forward``'s output layout.
            manifold_logits: ``(..., K)`` raw gating logits.

        Returns:
            loss: scalar, averaged over every non-branch dimension.
            gate: ``(..., K)`` gating probabilities.
        """
        if per_branch_losses.shape[0] != manifold_logits.shape[-1]:
            raise ValueError(
                f"expected per-branch losses with leading axis "
                f"{manifold_logits.shape[-1]}, got {tuple(per_branch_losses.shape)}"
            )
        gate = manifold_logits.softmax(dim=-1)
        weights = gate.movedim(-1, 0)                 # (K, ...)
        return (weights * per_branch_losses).sum(dim=0).mean(), gate


class ManifoldModelFrameworkV2(ManifoldModelFramework):
    def __init__(self, manifolds, encoder, decoder, encoder_out, decoder_in):
        super().__init__(manifolds, encoder, decoder, encoder_out, decoder_in)
        if len(manifolds) == 0:
            raise ValueError(f"Number of target manifolds must be >= 1, got 0")
        if encoder is None or decoder is None:
            raise ValueError(f"Encoder and decoder must not be None")
        if encoder_out < 1 or decoder_in < 1:
            raise ValueError(
                f"encoder_out and decoder_in must be >= 1, got {encoder_out} and {decoder_in}"
            )
        # self.manifolds = manifolds
        # self.num_manifolds = len(manifolds)
        # self.encoder = encoder
        # self.decoder = decoder
        # self.encoder_out = encoder_out
        # self.decoder_in = decoder_in
        
        self.to_ambient = nn.ModuleList(
            nn.Linear(self.encoder_out, m.ambient_dim) for m in manifolds
        )
        sum_ambient = sum([m.ambient_dim for m in self.manifolds])
        self.from_ambient = nn.Linear(sum_ambient, self.decoder_in)

    
    def forward(self, x):
        x = self.encoder(x)
        outputs = []
        for manifold, to_ambient in zip(
            self.manifolds, self.to_ambient
        ):
            z = to_ambient(x)
            z = manifold.project(z)
            outputs.append(z)
        
        z = torch.cat(outputs, dim=-1)
        z = self.from_ambient(z)
        return self.decoder(z)