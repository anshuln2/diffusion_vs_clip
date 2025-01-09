import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import open_clip
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torchvision
import wandb
# from diffusers.model

import torch
from torchvision import datasets, transforms
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 32

# Define the color map (e.g., 10 colors for 10 classes)
colors = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [128, 0, 0],    # Maroon
    [128, 128, 0],  # Olive
    [0, 128, 0],    # Green
    [0, 128, 128]   # Teal
]

# Get random characters for labels
label_characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
color_characters = ['R', 'G', 'B', 'Y', 'C', 'M', 'M', 'O', 'G', 'T']

# Load MNIST dataset
transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                ])
mnist_train = datasets.MNIST(root='.', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='.', train=False, transform=transform, download=True)


def colorize_image(img, color):
    # Convert the grayscale image to a colored image
    # img = img.squeeze(0).numpy()
    # img_new = 1. - img
    # img_colored = np.stack([img_new * color[i] / 255 for i in range(3)], axis=0)
    # img_colored = img_colored + np.expand_dims(img_new, 0)
    # img_colored = np.clip(img_colored, 0, 1)
    img = img.squeeze(0).numpy()
    background_img = np.ones((3, 32, 32))
    img_colored = np.stack([background_img[i] * color[i] / 255 for i in range(3)], axis=0)
    img_colored = img_colored + np.expand_dims(img, 0)
    img_colored = np.clip(img_colored, 0, 1)
    return torch.tensor(img_colored, dtype=torch.float32)

def reverse_diffusion_step(noisy_image, model, noise_scheduler, t):
    # Predict the noise to be removed
    noise_pred = model(noisy_image, t, return_dict=False)[0]
    
    denoised_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample    
    # # Compute the next step in the reverse diffusion process
    # alpha_t = noise_scheduler.alphas[t]
    # alpha_t_prev = noise_scheduler.alphas[t - 1] if t > 0 else 1.0
    
    # # Compute the denoised image (simplified example)
    # denoised_image = (noisy_image - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    
    # # Add some noise for the next step
    # if t > 0:
    #     noise = torch.randn_like(noisy_image).cuda()
    #     denoised_image += (alpha_t_prev - alpha_t).sqrt() * noise
    
    return denoised_image

def visualize(model, noise_scheduler, bsz=1):
    with torch.no_grad():
        model.eval()
                # Generate initial noise
        image_shape = (bsz, 3, 32, 32)  # Define the shape of your images
        initial_noise = torch.randn(image_shape).cuda()
        
        # Reverse diffusion process
        timesteps = torch.arange(999, 0, -1).cuda()

        for t in tqdm(timesteps):
            initial_noise = reverse_diffusion_step(initial_noise, model, noise_scheduler, t)

        out = torch.clamp(initial_noise, 0, 1)
        return out
    

def create_colored_dataset(dataset, ds_size=1000000):
    colored_dataset = []
    for idx, (img, label) in enumerate(dataset):
        if idx >= ds_size:
            break
        color = colors[label < 5]
        img_colored = colorize_image(img, color)
        prompt = f"{label_characters[label]} {color_characters[int(label < 5)]}"
        colored_dataset.append({'image': img_colored, 'label': label, 'color': color, 'prompt': prompt})
    return colored_dataset

# Create colored MNIST datasets
colored_mnist_train = create_colored_dataset(mnist_train, ds_size=BATCH_SIZE*100000)
colored_mnist_test = create_colored_dataset(mnist_test)

train_dataloader = torch.utils.data.DataLoader(colored_mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(colored_mnist_test, batch_size=BATCH_SIZE, shuffle=False)


model = UNet2DModel(
    sample_size=32,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
).cuda()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # , beta_schedule=args.ddpm_beta_schedule)
optimizer = torch.optim.AdamW(
    model.parameters(), #  +[x for x in text_encoder.parameters()],  # Add CLIP params here
    lr=1e-4
    # lr=args.learning_rate,
    # betas=(args.adam_beta1, args.adam_beta2),
    # weight_decay=args.adam_weight_decay,
    # eps=args.adam_epsilon,
)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_steps,
#     num_training_steps=(len(train_dataloader) * config.num_epochs),
# )
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000000, eta_min=1e-4)
from tqdm import tqdm

NUM_TRAIN_STEPS = 10000
LOGGING_FREQUENCY = 10
VAL_FREQUENCY = 50 # How often to produce images for visual validation
# BATCH_SIZE = 128

curr_loss = 0
train_step = 0
wandb_run = wandb.init(project="diffusion_training_cmnist", config={'batch_size': BATCH_SIZE, 'lr': 1e-4, 'model': 'unet2d'})

for epoch in range(10000):
    for batch in train_dataloader:

        train_step += 1
        images = batch["image"].to(torch.float32).cuda()
        bsz = images.shape[0]
        # Get noise
        noise = torch.randn_like(images)
        # Get a random timestep
        timesteps = torch.randint(0, 1000, (bsz,), device=noise.device)
        timesteps = timesteps.long()


        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).cuda()

        
        model_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = torch.nn.functional.mse_loss(model_pred, noise)  # this could have different weights!

        loss.backward()
        
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        curr_loss += loss.item()
        if train_step % LOGGING_FREQUENCY == 0:
            # Will add to wandb
            print(f"Step {train_step}: Loss {curr_loss/LOGGING_FREQUENCY}")
            curr_loss = 0
            wandb_run.log({"loss": loss.item(), "step": train_step})
        if train_step % VAL_FREQUENCY == 0:
            # Visualize
            out = visualize(model, noise_scheduler)
            torchvision.utils.save_image(out, f"outputs/out_{train_step}.png")
            wandb_run.log({"image": wandb.Image(out)})        
    # print(f"Epoch {epoch} done")