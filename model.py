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
