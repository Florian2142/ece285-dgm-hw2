"""The GAN objective: a game between two networks.   >>> TODOs are your homework. <<<

Goodfellow et al. (2014), with D(x) = sigmoid(logit) the probability that x is real:

    min_G max_D  V(D, G) = E_{x ~ p_data}[ log D(x) ]  +  E_{z ~ p(z)}[ log(1 - D(G(z))) ]

Three versions of the game (``cfg.train.loss``):

- ``"minimax"``        D maximizes V, G minimizes V (the game above, exactly)
- ``"nonsaturating"``  D as in minimax; G maximizes E_z[ log D(G(z)) ] instead (same paper, Sec. 3)
- ``"hinge"``          D minimizes E_x[ max(0, 1 - l(x)) ] + E_z[ max(0, 1 + l(G(z))) ],
                       G minimizes -E_z[ l(G(z)) ], with l = D's logit (Lim & Ye, 2017;
                       Miyato et al., 2018)

Conventions that train.py, the plots, and the report rely on: both functions take LOGITS,
return ONE number (expectations = batch means), and that number is MINIMIZED by the
optimizer of its network.

Work through the TODOs in this order and check each one as you go:

    1. discriminator_loss    uv run pytest homeworks/hw2 -k dloss
    2. generator_loss        uv run pytest homeworks/hw2 -k gloss
"""

from __future__ import annotations

from torch import Tensor
import torch

LOSSES = ("minimax", "nonsaturating", "hinge")


def discriminator_loss(
    real_logits: Tensor, fake_logits: Tensor, kind: str = "nonsaturating"
) -> Tensor:
    """The loss D minimizes; ``real_logits [B]`` = D(x), ``fake_logits [B]`` = D(G(z)); scalar."""

    ########################################################################
    # TODO: All three kinds; raise a ValueError for any other name.        #
    #                                                                      #
    # For "minimax" and "nonsaturating", D maximizes V: write -V as a loss #
    # (the two expectations are two batch means). Which classification     #
    # loss is that, with which targets for real and fake?                  #
    #                                                                      #
    # Hint: work with the logits. Why is torch.log(torch.sigmoid(l))       #
    # unsafe for l = -100, and which torch function avoids it?             #
    ########################################################################

    # non-saturating and minmax are the same here in this case since its the discriminator
    if kind in ["minimax", "nonsaturating"]:
        loss = torch.nn.functional.softplus(-real_logits).mean() + torch.nn.functional.softplus(fake_logits).mean()
    elif kind == "hinge":
        # we do not really need it honestly but its nice to have
        loss = torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean()
    else:
        raise ValueError


    # return the loss as typical
    return loss
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


def generator_loss(fake_logits: Tensor, kind: str = "nonsaturating") -> Tensor:
    """The loss G minimizes; ``fake_logits [B]`` = D(G(z)); scalar."""

    ########################################################################
    # TODO: All three kinds; raise a ValueError for any other name.        #
    #                                                                      #
    # "minimax": G minimizes the part of V that depends on G.              #
    # "nonsaturating": G maximizes E_z[ log D(G(z)) ].                     #
    #                                                                      #
    # Question for you: early in training D spots the fakes easily, so     #
    # D(G(z)) is close to 0. Which of the two gives G a useful gradient    #
    # then? (Notebook 02, section 1 plots both.)                           #
    ########################################################################

    if kind == 'minimax': # the saturating loss which becomes a problem over time
        loss = -torch.nn.functional.softplus(fake_logits).mean()
    elif kind == 'nonsaturating': # the better formula
        loss = torch.nn.functional.softplus(-fake_logits).mean() # negative because pytorch like typical always minimizes
    elif kind == 'hinge': # again same as above
        loss = -fake_logits.mean()
    else: # if nohting was choosen raise Valueerror
        raise ValueError

    return loss
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
