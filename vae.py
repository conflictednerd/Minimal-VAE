from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Vanilla_VAE(nn.Module):
    """
    Vanilla VAE. Operates on a batch of vectors. Uses fully connected layers.
    Can easily be changed so that encoder/decoder use different architectures (Conv net, Recurrent net, etc.) or handle different input shapes (e.g., 2d images, time-series)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [256, 256, 256],
        optim_param: Dict = {"lr": 3e-4},
        beta: float = 1.0,
    ):
        """
        Args:
            input_dim: input dimension. Inputs should be 1d vectors of this length.
            latent_dim: dimensionality of vae's latent space
            hidden_dim: list of hidden layers used in both the encoder and the decoder.
            optim_param: Parameters passed to the optimizer, in particular, the learning rate
            beta: coefficient of KL part in vae loss
        """
        super().__init__()
        self.encoder = Vanilla_Encoder(
            in_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        self.decoder = Vanilla_Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            out_dim=input_dim,
        )
        self.beta = beta
        self.device = device
        self._configure_optimizer(optim_param)

    def _configure_optimizer(self, optim_params: Dict):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=optim_params["lr"])

    def reconstruction_loss(self, p: torch.distributions.Distribution, x: torch.Tensor):
        """Returns the reconstruction loss for a batch of inputs
        Args:
            p: likelihood distribution approximated by the decoder
            x: (ground truth) inpupts

        Returns:
            reconstruction loss to be minimized. This is the negative log likelihood of x under p.
            shape is (B, )
        """
        logp = p.log_prob(x)
        # Regardless of what the shape is, returns a 1d tensor of shape (B)
        return -logp.sum(dim=tuple(range(1, logp.ndim)))

    def kl_loss(
        self,
        q: torch.distributions.Distribution,
        z: torch.Tensor = None,
        method="analytical",
    ):
        """Returns the KL divergence between the posterior and the prior.
        The prior is assumed to be standard Gaussian.
        Args:
            q: posterior over the latents approximated by the encoder
            z: a sample from q obtained via reparameterization
            method: One of 'analytical' (default choice), 'monte-carlo', 'schulman'

        Returns:
            kl divergence to be minimized.
            shape is (B, )
        """
        assert method in [
            "analytical",
            "monte-carlo",
            "schulman",
        ], "KL calculation method was not found"
        if method != "analytical":
            assert (
                z is not None
            ), "For the estimation of KL, you need to pass in a sample from the distribution q"
        prior = torch.distributions.Normal(
            torch.zeros_like(q.mean), torch.ones_like(q.stddev)
        )
        if method == "monte-carlo":
            logp_q = q.log_prob(z)
            logp_p = prior.log_prob(z)
            kl = logp_q - logp_p
        elif method == "schulman":
            logp_q = q.log_prob(z)
            logp_p = prior.log_prob(z)
            kl = 0.5 * (logp_q - logp_p) ** 2
        else:
            kl = torch.distributions.kl.kl_divergence(q, prior)

        return kl.sum(-1)

    def forward(self, x):
        """
        Expects well-formed inputs of shape (Bxn)
        Returns:
            elbo: The scalar loss that you should call backwards on  
            info: A dictionary containing additional information
        """
        q = self.encoder(x)  # approximate posterior
        z = q.rsample()  # latent sample (sampled with reparameterization)
        p = self.decoder(z)  # approximate likelihood

        recon_loss = self.reconstruction_loss(p, x)
        kl_loss = self.kl_loss(q, z)

        elbo = (self.beta * kl_loss + recon_loss).mean()
        with torch.no_grad():
            info = {
                "elbo": elbo.item(),
                "kl": kl_loss.mean().item(),
                "recon_loss": recon_loss.mean().item(),
            }
        return elbo, info


class Vanilla_Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(m, n) for m, n in zip([in_dim] + hidden_dims, hidden_dims)]
        )
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        mu = self.mu(x)
        std = torch.exp(self.logvar(x) / 2)
        q = torch.distributions.Normal(mu, std)  # q(z|x): approximate posterior
        return q


class Vanilla_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(m, n)
                for m, n in zip([latent_dim] + hidden_dims, hidden_dims + [out_dim])
            ]
        )
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        mu = self.layers[-1](x)
        std = torch.exp(self.log_scale)
        p = torch.distributions.Normal(mu, std)  # p(x|z): approximate likelihood
        return p