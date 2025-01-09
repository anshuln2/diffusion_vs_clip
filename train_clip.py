import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import open_clip
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# import transformers
from open_clip.transformer import VisionTransformer
# from diffusers.model

import torch
from torchvision import datasets, transforms
import numpy as np
import random
import wandb

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
label_characters = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K']
color_characters = ['R', 'G', 'B', 'Y', 'C', 'M', 'M', 'O', 'G', 'T']

LOSS_FN = 'siglip'
VISION_TOWER = 'vit'
LR = 5e-4
TRAIN_SIGNAL = 'label_color'

# Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                ])
mnist_train = datasets.MNIST(root='.', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='.', train=False, transform=transform, download=True)


def colorize_image(img, color):
    # Convert the grayscale image to a colored image
    img = img.squeeze(0).numpy()
    background_img = np.ones((3, 32, 32))
    img_colored = np.stack([background_img[i] * color[i] / 255 for i in range(3)], axis=0)
    img_colored = img_colored + np.expand_dims(img, 0)
    img_colored = np.clip(img_colored, 0, 1)
    return torch.tensor(img_colored, dtype=torch.float32)

def create_colored_dataset(dataset, ds_size=1000000):
    colored_dataset = []
    for idx, (img, label) in enumerate(dataset):
        if idx >= ds_size:
            break
        color = colors[label < 5]
        img_colored = colorize_image(img, color)
        prompt = f"{label_characters[label]} {color_characters[int(label < 5)]}"
        colored_dataset.append({'image': img_colored, 'label': label_characters[label], 'color': color_characters[int(label < 5)], 'prompt': prompt})
    return colored_dataset

# Create colored MNIST datasets
from collections import defaultdict
# Check if model understands shapes and colors

def create_inverted_colored_dataset(dataset):
    # Has different colors than above
    colored_dataset = []
    for img, label in dataset:
        color = colors[label >= 5]
        img_colored = colorize_image(img, color)
        prompt = f"{label_characters[label]} {color_characters[int(label >= 5)]}"
        colored_dataset.append({'image': img_colored, 'label': label_characters[label], 'color': color_characters[int(label >= 5)], 'prompt': prompt})
    return colored_dataset

# Also look at accuracy on original MNIST test set

colored_mnist_train = create_colored_dataset(mnist_train)
colored_mnist_test = create_colored_dataset(mnist_test)
colored_mnist_inverted = create_inverted_colored_dataset(mnist_test)

train_dataloader = torch.utils.data.DataLoader(colored_mnist_train, batch_size=64, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(colored_mnist_test, batch_size=64, shuffle=False)
inverted_dataloader = torch.utils.data.DataLoader(colored_mnist_inverted, batch_size=64, shuffle=False)

# And accuracy on color only images

# Assuming you have a tokenizer and your dataloader set up
class LabelEncoder:
    def __init__(self):
        self.prompt_to_label = {}
        self.label_count = 0
    
    def encode(self, prompts):
        labels = []
        for prompt in prompts:
            if prompt not in self.prompt_to_label:
                self.prompt_to_label[prompt] = self.label_count
                self.label_count += 1
            labels.append(self.prompt_to_label[prompt])
        return torch.tensor(labels).cuda()

def nt_xent_loss(image_features, text_features, labels, temperature=1.0):
    # Normalize features to unit length
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # Compute cosine similarity as dot product in feature space (batch_size x batch_size)
    logits = torch.matmul(image_features, text_features.t()) / temperature
    
    # Labels used to determine which entries are positives
    # Create a mask of positive samples: 1 for positives, 0 for negatives
    batch_size = labels.size(0)
    labels = labels.unsqueeze(1) # Shape: [batch_size, 1]
    mask = torch.eq(labels, labels.T).float() # Shape: [batch_size, batch_size]
    
    loss_img = -1.* torch.nn.functional.log_softmax(logits, dim=1) * (mask / mask.sum(1, keepdim=True))
    loss_txt = -1.* torch.nn.functional.log_softmax(logits, dim=0) * (mask / mask.sum(0, keepdim=True))
    loss = (loss_img.sum() + loss_txt.sum()) / (2 * batch_size)
    
    return loss
    

    
def siglip_loss(image_features, text_features, labels, logit_bias, t_prime):
    # Normalize features to unit length
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    t = torch.exp(t_prime)
    
    # Compute cosine similarity as dot product in feature space (batch_size x batch_size)
    logits = torch.matmul(image_features, text_features.t())  
    logits = logits * t + logit_bias   
    # sig_logits = torch.sigmoid(logits)
    mask = 2 * torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float() - 1 # Shape: [batch_size, batch_size]

    loss = -1.* torch.nn.functional.logsigmoid(logits * mask).sum() / mask.shape[0]
    return loss    

def test_model(test_loader, image_encoder, text_encoder, tokenizer, class_names, attribute):
    image_encoder.eval()
    text_encoder.eval()
    
    text = tokenizer(class_names).cuda()
    text_features = text_encoder(text)
    text_features = pool_text_features(text_features, text)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    get_class_idx = lambda x: class_names.index(x)
    
    correct = 0
    total = 0
    
    for batch in test_loader:
        image = batch['image'].cuda()
        if attribute == 'color':
            label = batch['color']
        elif attribute == 'shape':
            label = batch['label']
        else:
            raise ValueError("Incorrect attribute specified")
        image_features = image_encoder(image).float()
        image_features = F.normalize(image_features, p=2, dim=-1)        
        similarity_matrix = torch.matmul(text_features, image_features.T)
        predictions = torch.argmax(similarity_matrix, dim=0)
        class_idx = torch.tensor([get_class_idx(x) for x in label]).cuda()
        correct += torch.sum(predictions == class_idx).item()
        total += len(label)
    return correct / total
    
label_encoder = LabelEncoder()

total_loss = 0
train_step = 0

def pool_text_features(text_features, prompt):
    prompt_mask = prompt > 0 # torch.tensor([x > 0 for x in prompt]).to(text_features.device).float()
    text_features = text_features * prompt_mask.unsqueeze(2)
    text_features = text_features.sum(1) / prompt_mask.sum(1).unsqueeze(1)
    return text_features

tokenizer = open_clip.get_tokenizer('ViT-B-32') 
clip = open_clip.create_model('ViT-B-32')
text_encoder = torch.nn.Sequential(clip.token_embedding, clip.transformer, clip.ln_final).cuda()

if VISION_TOWER == 'vit':
    image_encoder = VisionTransformer(image_size=32,
                                        patch_size=4,
                                        layers=4,
                                        heads=1,
                                        width=256,
                                        mlp_ratio=2,
                                        output_dim=512).cuda()
elif VISION_TOWER == 'ffn':
    image_encoder = torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Linear(32*32*3, 512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512, 512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512, 512)
                    ).cuda()

optimizer = torch.optim.AdamW(
    [x for x in image_encoder.parameters()]+[x for x in text_encoder.parameters()],  # Add CLIP params here
    lr=LR
    # lr=args.learning_rate,
    # betas=(args.adam_beta1, args.adam_beta2),
    # weight_decay=args.adam_weight_decay,
    # eps=args.adam_epsilon,
)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50000, eta_min=1e-4)

wandb_run = wandb.init(project='clip_cmnist', config={'lr': LR, 'batch_size': 64, 'loss': LOSS_FN, 'train_signal': TRAIN_SIGNAL, 'image_tower': VISION_TOWER})

temperature = torch.nn.Parameter(torch.tensor(np.log(10)))
logit_bias = torch.nn.Parameter(torch.tensor(-10.))

SAVE_FREQUENCY = 1000


for epoch in range(200):
    image_encoder.train()
    text_encoder.train()
    for batch in train_dataloader:
        train_step += 1
        image = batch['image'].cuda()
        label = batch['label']
        prompt = batch['prompt']
        text = tokenizer(prompt).cuda()
        text_features = text_encoder(text)
        image_features = image_encoder(image).float()
        # print(image_features.shape)
        enc_labels = label_encoder.encode(prompt)
        # print(text_features.shape)
        if LOSS_FN == 'info_nce':
            loss = nt_xent_loss(image_features, pool_text_features(text_features, text), enc_labels)
        elif LOSS_FN == 'siglip':
            loss = siglip_loss(image_features, pool_text_features(text_features, text), enc_labels, logit_bias, temperature)
        # loss = nt_xent_loss(image_features, pool_text_features(text_features, text), enc_labels)
        # Compute the loss
        # We use contrastive loss like CLIP
        # The contrastive loss is computed between the text and image features
        # The positive pairs are the text-image pairs with the same label
        # The negative pairs are the text-image pairs with different labels
        # The loss is minimized when the positive pairs are closer than the negative pairs
        # The margin is used to separate the positive and negative pairs
        # margin = 0.1
        # similarity_matrix = torch.matmul(text_features, image_features.T)
        # positive_pairs = torch.diag(similarity_matrix[label])
        # negative_pairs = similarity_matrix - torch.eye(similarity_matrix.size(0)).cuda() * 1000
        # negative_pairs = torch.max(negative_pairs, dim=1).values
        # loss = torch.nn.functional.relu(negative_pairs - positive_pairs + margin).mean()
        # loss = siglip_loss(image_features, pool_text_features(text_features, text), enc_labels, logit_bias, temperature)
        # Now we optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()
        total_loss += loss.item()
        if train_step % 50 == 0:
            print(f"Step {train_step}: Loss {total_loss/50}")
            wandb_run.log({'loss': total_loss/50, 'step': train_step})
            total_loss = 0
        if train_step % SAVE_FREQUENCY == 0:
            final_state_dict = {'vision': image_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'text_encoder': text_encoder.state_dict()}
            torch.save(final_state_dict, f"outputs/clip_cmnist.pt")   
    image_encoder.eval()
    text_encoder.eval()
    color_acc_orig = test_model(test_dataloader, image_encoder, text_encoder, tokenizer, color_characters[:2], attribute='color')
    shape_acc_orig = test_model(test_dataloader, image_encoder, text_encoder, tokenizer, label_characters, 'shape')
    color_acc_inv = test_model(inverted_dataloader, image_encoder, text_encoder, tokenizer, color_characters[:2], 'color')
    shape_acc_inv = test_model(inverted_dataloader, image_encoder, text_encoder, tokenizer, label_characters, 'shape')
    # print(f"Color accuracy on original dataset: {color_acc_orig}")
    # print(f"Shape accuracy on original dataset: {shape_acc_orig}")
    # print(f"Color accuracy on inverted dataset: {color_acc_inv}")
    # print(f"Shape accuracy on inverted dataset: {shape_acc_inv}")
    wandb_run.log({'color_acc_orig': color_acc_orig,
                   'shape_acc_orig': shape_acc_orig,
                   'color_acc_inv': color_acc_inv, 'shape_acc_inv': shape_acc_inv,
                   'step': train_step})
    