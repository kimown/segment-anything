import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

checkpoint = os.path.join("model/sam_vit_l_0b3195.pth")
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)


image = cv2.imread('src/assets/data/dogs.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("src/assets/data/dogs_embedding.npy", image_embedding)
