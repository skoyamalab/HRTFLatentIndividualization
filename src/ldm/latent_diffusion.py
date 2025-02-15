import torch
from icecream import ic

class DDIMScheduler:
    def __init__(self, num_train_timesteps=3000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device='cuda')
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)

    def step(self, model_output, timestep, sample, eta=0.0):
        # Compute alpha_prod_t and alpha_prod_t_prev
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)

        # Compute predicted x0
        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # Compute "direction pointing to x_t"
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - eta * variance) * model_output

        # Compute x_t-1
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
def train_step(model, scheduler, optimizer, latents, condition_embeddings, criterion, max_grad_norm=1.0, condition_dropout_prob=0.1):
    # optimizer.zero_grad()

    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

    # Generate time embeddings
    time_embeddings = model.time_step_embedding(timesteps)

    # Generate random noise
    noise = torch.randn_like(latents)

    # Add noise to the latents
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # ic(noisy_latents.shape, latents.shape)

    # Randomly decide whether to use the condition
    use_condition = torch.rand(1).item() > condition_dropout_prob

    # Predict the noise
    noise_pred = model(noisy_latents, time_embeddings, condition_embeddings if use_condition else None)

    # Compute loss
    loss = criterion(noise_pred, noise)

    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return loss.item()

def test_step(model, scheduler, latents, condition_embeddings, criterion, guidance_scale=7.5):
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

    # Generate time embeddings
    time_embeddings = model.time_step_embedding(timesteps)

    with torch.no_grad():
        # Generate random noise
        noise = torch.randn_like(latents)

        # Add noise to the latents
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # # Predict the noise
        # noise_pred = model(noisy_latents, time_embeddings, condition_embeddings)

        # Get unconditional prediction
        noise_pred_uncond = model(noisy_latents, time_embeddings, None)
        
        # Get conditional prediction
        noise_pred_cond = model(noisy_latents, time_embeddings, condition_embeddings)
        
        # Combine predictions using classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute loss
        loss = criterion(noise_pred, noise)

    return loss.item()

def test_generation(model, scheduler, condition_embeddings, guidance_scale=7.5):
    # Generate random latents
    denoised_latents = torch.randn((1, 1, 256, 64), device=condition_embeddings.device)

    with torch.no_grad():
        # Iterate over timesteps in reverse order
        for t in scheduler.timesteps:
            time_embeddings = model.time_step_embedding(torch.tensor([t], device=denoised_latents.device))

            # Get unconditional prediction
            noise_pred_uncond = model(denoised_latents, time_embeddings, None)
            
            # Get conditional prediction
            noise_pred_cond = model(denoised_latents, time_embeddings, condition_embeddings)
            
            # Combine predictions using classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            denoised_latents = scheduler.step(noise_pred, t, denoised_latents, eta=0.5)

    return denoised_latents