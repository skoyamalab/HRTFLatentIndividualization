import torch
import math


class DDIMScheduler:
    def __init__(
        self, device, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02
    ):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, device=self.device
        )
        # self.betas = self.cosine_beta_schedule(self.num_train_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device) / timesteps
        alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )

    def step(self, model_output, timestep, sample, eta=0.0):
        # Compute alpha_prod_t and alpha_prod_t_prev
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep - 1]
            if timestep > 0
            else torch.tensor(1.0).to(self.device)
        )

        # Compute predicted x0
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)
        pred_original_sample = torch.clamp(pred_original_sample, -3.0, 3.0)

        # Compute variance
        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

        # Compute "direction pointing to x_t"
        pred_sample_direction = (
            torch.sqrt(1 - alpha_prod_t_prev - eta * variance) * model_output
        )

        # Compute x_t-1
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[
            timesteps
        ].flatten()

        # Ensure coefficients are shaped correctly for broadcasting
        # Original shape: (Batch,)
        # Target shape for 1D data (B, C, L): (Batch, 1, 1)
        # Target shape for 2D data (B, C, H, W): (Batch, 1, 1, 1)

        # Determine required shape based on input dimensions
        num_dims_to_add = (
            original_samples.ndim - 1
        )  # Need trailing dims for C, L (or C, H, W)

        view_shape = [-1] + [1] * num_dims_to_add

        sqrt_alpha_prod = sqrt_alpha_prod.view(view_shape)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(view_shape)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples
