# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
model_path = "/home/morand/Downloads/FIVESA/QTSeg/FPNEncoderMaskDecoder/20250302-103916/weight_best_iou.pt"

# %%
model = torch.load(model_path, map_location=torch.device('cpu'))

# %%
print(model.keys())

# %%
import json

json_path = "/home/morand/Downloads/FIVESA/QTSeg/FPNEncoderMaskDecoder/20250302-103916/cfg.json"


with open(json_path, 'r') as f:
    config = json.load(f)

print(config)

# %%
from torchinfo import summary

summary(model)

# %%
