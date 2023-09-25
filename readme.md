# Minimal VAE

This is a minimal implementation of variational autoencoder (VAE) in pytorch.

I have tried to make this code as close to the underlaying math as possible, while retaining maximum flexibility. Because of this, it should be very easy to change the architecture of encoder/decoder, operate on other modalities of data like images instead of flattened vectors, modify the loss functions, etc.

A VAE is a generative model that can approximate complex distributions using only samples from that distribution.
Using variational inference methods, we can leverage a dataset of samples to fit the parameters of a latent variable model. Latent variable models are very powerful as they can approximate many complex distributions. Therefore, they have become a popular choice for learning complex, high-dimensional data distributions, such as images, trajectories, etc.

## About This Code

In `vae.py` you can find a very minimal and clean implementation of a VAE. The code is written such that you can easily adapt it to your use cases without too much refactoring. It contains three main classes: `Vanilla_Encoder`, `Vanilla_Decoder` and, `Vanilla_VAE`.

The `Vanilla_Encoder` and `Vanilla_Decoder` classes are regular `nn.Modules` that implement the encoder and decoder of a VAE. Noticeably, they each output a `torch.distributions.Distribution` object, in line with the theory of variational autoencoders.

A VAE's loss function (known as - evidence lower bound or ELBO) is comprised of two main components: a reconstruction loss and a KL-divergence.

$$\mathcal{L}(x) = \underbrace{D_{\mathrm{KL}} (q(z|x)||p(z))}_{\text{KL term}} - \underbrace{\mathbb{E}_{z\sim q(z|x)}[\log p(x|z)]}_{\text{Reconstruction loss}}$$

Given an input sample $x$, it is first passed through the encoder which returns a distribution $q(z|x)$ over the latent variables. ($q(z|x)$ is called the *approximate posterior*) This distribution should be close to a prior distribution that we have specified. The KL divergence between the approximate posterior and the prior, $p(z)$ gives us the KL term in VAE's loss function. To get the reconstruction term, we use the reparameterization trick to get a differentiable sample $z$ from the distribution $q(z|x)$. We pass this sample to the decoder which outputs another distribution, $p(x|z)$. (It is called the conditional or the likelihood distribution) The reconstruction loss is basically the log-likelihood of the original data point $x$, under this distribution. By maximizing this, we are incentivizing the decoder to accurately reconstruct $x$ from a sampled latent $z$.

**Note 1:** In many VAE tutorials, the reconstruction loss is written as an MSE loss between the original data point $x$ and the output of the decoder (which is considered to be a tensor, rather than a distribution). This corresponds to the case of Gaussian $p(x|z)$, where the decoder outputs the mean of this distribution. In general, we can let the output of the decoder be any distribution with a differentiable log-likelihood. In this code, the decoder outputs the full distribution object and the reconstruction loss is implemented as the log-likelihood of it. As a result, if you want to use a non-Gaussian $p(x|z)$ all you have to do is to change the `Vanilla_Decoder` class so that it outputs the distribution that you like. The rest of the code does not need any changes.

The same is true for the approximate posterior $q(z|x)$: If you want to use another distribution, you only need to change the `Vanilla_Encoder` class so that it computes and outputs the appropriate distribution.

**Note 2:** The prior distribution $p(z)$ is assumed to be the standard normal distribution. If you want to change it, you can edit the `kl_loss` method of `Vanilla_VAE`.

**Note 3:** `Vanilla_VAE` class provides two method `kl_loss()` and `reconstruction_loss()` to compute each part of the loss. In many cases, the KL divergence between $q(z|x)$ and the prior $p(z)$ can be computed analytically. However, we may want to use other distributions where this is not possible. In this case, we can use the sampled latent $z \sim q(z|x)$ (which is passed to the decoder) to estimate the KL term. In `kl_loss()` you can choose between a monte-carlo estimate which is an unbiased estimator of KL or another estimator proposed by John Schulman which is biased, but has considerably lower variance. (see [here](http://joschu.net/blog/kl-approx.html) for more information about these estimators)



By passing a batch of data to the forward method of a `Vanilla_VAE` object, you get a scalar loss and a dictionary containing additional information. Training the VAE is therefore very easy. You can loop over batches of data and update the VAE:

```python
for data in data_loader:
    loss, info = vae(data)
    vae.optimizer.zero_grad()
    loss.backward()
    vae.optimizer.step()
```

Given a batch of data, If you want to get their latent representation, you can use the following line of code:

```python
latent_representation = vae.encoder(data).mean
```

And if you want to get their reconstructions (e.g., for visualization), you can use

```python
reconstructed = vae.decoder(latent).mean
```

