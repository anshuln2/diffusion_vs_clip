# Diffusion v/s CLIP
This repo aims to understand how diffusion models and CLIP-style models learn different features from the same data.

I use ColoredMNIST data to train a CLIP style model (in `train_clip.py`) and to train conditional and unconditional diffusion models. I then try to see if these models generalize to different color-shape combinations.
