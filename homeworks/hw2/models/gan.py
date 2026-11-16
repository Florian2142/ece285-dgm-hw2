"""Your GAN for 3 x H x W paintings.   >>> The TODO blocks in this file are your homework. <<<

Same structure as models/vae.py in HW1: two networks, and a ``GAN`` that holds them.

    z [B,K] ~ N(0, I) --Generator--> x_fake [B,3,H,W] --+
                                                         +--Discriminator--> logits [B]
                                  x_real [B,3,H,W] --+

The two players:

- generator      G_theta(z): noise -> image                     (parameters theta)
- discriminator  D_phi(x) = sigmoid(logit): P(x is a real image) (parameters phi)

Work through the TODOs in this order and check each one as you go
(skipped = still TODO, passed = done; or use notebooks/02_gan_step_by_step.ipynb):

    1. Generator          uv run pytest homeworks/hw2 -k generator
    2. Discriminator      uv run pytest homeworks/hw2 -k discriminator
    3. init_weights       uv run pytest homeworks/hw2 -k init
    4. GAN.sample         uv run pytest homeworks/hw2 -k sample

Contract (train.py, evaluate.py, and the tests use only this interface). Shapes for a batch of
B images and latent dimension K = cfg.latent_dim:

- images inside the GAN: float ``[B, 3, H, W]``, values in [-1, 1], ``H = W = image_size``
  (train.py maps the data from [0, 1] with ``to_model_range``)
- ``Generator(z)`` -> ``[B, 3, H, W]`` in [-1, 1]
- ``Discriminator(x)`` -> ``logits [B]``: unconstrained, no sigmoid (``losses.py`` uses them)
- ``GAN.sample(n)`` -> ``[n, 3, H, W]`` in [0, 1]: NEW images for image grids and FID
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hw2.config import ModelConfig

# for the training run in order to map the picture on same dim like -1,1
def to_model_range(x: Tensor) -> Tensor:
    """Images in [0, 1] (data loader) -> [-1, 1] (the GAN)."""
    return 2 * x - 1

# We need this for later evaluating the pictures for the FID and others
def to_image_range(x: Tensor) -> Tensor:
    """Images in [-1, 1] (the GAN) -> [0, 1] (image grids, FID)."""
    return ((x + 1) / 2).clamp(0, 1)


class Generator(nn.Module):
    """``G_theta``: a latent code ``z [B, K]`` -> an image ``[B, 3, H, W]`` in [-1, 1]."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.image_size = image_size

        ########################################################################
        # TODO: Build the generator layers.                                    #
        #                                                                      #
        # Choices to make (and to justify in the report): how z becomes a      #
        # small spatial map, how H and W grow (image_size must be divisible    #
        # by 2**number_of_upsamplings), width (cfg.g_channels), activations,   #
        # normalization layers. DCGAN (Radford et al., 2016) is the classic    #
        # starting point.                                                      #
        #                                                                      #
        # Question for you: which last activation gives values in [-1, 1],     #
        # and why must the real images use the same range?                     #
        ########################################################################

        c = cfg.g_channels * 8        # for not hardcoding

        self.layers = nn.Sequential(
            nn.Linear(self.latent_dim, c * 4 * 4), # we take the 2016 paper on GAN'S -> Radford et al.
            nn.Unflatten(1, (c, 4,4)), # reshapes into deep image
            nn.BatchNorm2d(c),
            nn.ReLU(),

            nn.ConvTranspose2d(c, c//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c//2),
            nn.ReLU(),

            nn.ConvTranspose2d(c//2, c//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c//4),
            nn.ReLU(),

            nn.ConvTranspose2d(c//4, c//8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c//8),
            nn.ReLU(),

            nn.ConvTranspose2d(c//8, 3, kernel_size=4, padding=1, stride=2),
            nn.Tanh()


        )

        '''
        We take the structure from the paper:

        Linear + Unflatten    → [512,  4,  4]   + BN + ReLU
        block: 512 → 256      → [256,  8,  8]
        block: 256 → 128      → [128, 16, 16]
        block: 128 →  64      → [ 64, 32, 32]
        last:   64 →   3      → [  3, 64, 64]   + Tanh   (no BN here)


        '''

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, z: Tensor) -> Tensor:
        """``z [B, K]`` -> images ``[B, 3, H, W]`` in [-1, 1]."""

        ########################################################################
        # TODO: Feed z through your layers and return the images.              #
        ########################################################################

        x_hat = self.layers(z) # get the image 

        return x_hat
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


class Discriminator(nn.Module):
    """``D_phi``: an image ``[B, 3, H, W]`` in [-1, 1] -> one logit per image ``[B]``."""

    def __init__(self, cfg: ModelConfig, image_size: int,  leaky_relu_slope=0.2) -> None:
        super().__init__()
        self.image_size = image_size
        self.lr_slope = leaky_relu_slope # default as the original paper

        ########################################################################
        # TODO: Build the discriminator layers.                                #
        #                                                                      #
        # Usually the mirror image of your generator: from 3 x H x W down to   #
        # ONE number per image (cfg.d_channels). Do NOT end with a sigmoid:    #
        # return logits (see losses.py for why).                               #
        #                                                                      #
        # Question for you: DCGAN uses LeakyReLU in D but ReLU in G. What      #
        # would a plain ReLU in D do to the gradient that G receives?          #
        ########################################################################

        c = cfg.d_channels # the choosen depth for first layer

        self.layers = nn.Sequential(

            nn.Conv2d(3, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(self.lr_slope),

            nn.Conv2d(c, c*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(self.lr_slope),

            nn.Conv2d(c*2, c*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c*4),
            nn.LeakyReLU(self.lr_slope),

            nn.Conv2d(c*4, c*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c*8),
            nn.LeakyReLU(self.lr_slope),

            nn.Conv2d(c*8, 1, kernel_size=4, stride=1, padding=0) # we use formula ((H - F + 2*P)/S + 1) hence stride = 1 or 2 words here

        )
        '''
        Here again we use a mirrored image and strided convs to crunch down the size

        input                    → [  3, 64, 64]
        conv:   3 →  c   (64)    → [ 64, 32, 32]   LeakyReLU        (no BN here)
        conv:   c → 2c  (128)    → [128, 16, 16]   BN + LeakyReLU
        conv:  2c → 4c  (256)    → [256,  8,  8]   BN + LeakyReLU
        conv:  4c → 8c  (512)    → [512,  4,  4]   BN + LeakyReLU
        last:  [512, 4, 4] → 1 number per image    (no BN, no activation)
f
        '''

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: Tensor) -> Tensor:
        """``x [B, 3, H, W]`` -> ``logits [B]`` (no sigmoid here)."""

        ########################################################################
        # TODO: Feed x through your layers and return one logit per image.     #
        ########################################################################

        logits = self.layers(x) # pass through discmriniator

        logits = logits.view(-1)    # [B, 1, 1, 1] → [B] it wants one number


        return logits

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


def init_weights(module: nn.Module) -> None:
    """DCGAN initialization; ``GAN`` applies it to every submodule (``model.apply``)."""

    ########################################################################
    # TODO: As in DCGAN (Radford et al., 2016): weights of convolution,    #
    # transposed convolution, and linear layers ~ N(0, 0.02^2), biases 0;  #
    # BatchNorm weights ~ N(1, 0.02^2), biases 0. Leave other modules as   #
    # they are.                                                            #
    #                                                                      #
    # Hint: isinstance and the functions in torch.nn.init.                 #
    ########################################################################
    '''
    We use a special initialization of the weights given the paper.
    '''
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02) # N(0, 0.02^2)
        nn.init.zeros_(module.bias) # bias 0
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight,mean=1, std=0.02) # N(1, 0.02^2)
        nn.init.zeros_(module.bias)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


class GAN(nn.Module):
    """Generator + Discriminator. Training uses the two networks; ``sample`` makes new images."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.image_size = image_size
        self.generator = Generator(cfg, image_size)
        self.discriminator = Discriminator(cfg, image_size)
        if cfg.init == "dcgan":
            self.apply(init_weights)
        elif cfg.init != "default":
            raise ValueError(f"Unknown init: {cfg.init!r}")

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        """Generate ``n`` NEW images ``[n, 3, H, W]`` in [0, 1].

        Use ``next(self.parameters()).device`` when ``device`` is None.
        """

        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n, self.latent_dim, device=device)
        return to_image_range(self.generator(z))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
