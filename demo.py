import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn.state import torch_load, load_state_dict
from PIL import Image
import numpy as np

from models.croco import CroCoNet

ckpt = torch_load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth')
model = CroCoNet()
load_state_dict(model, ckpt['model'], strict=True)


imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
imagenet_mean_tensor = tinygrad.Tensor(imagenet_mean)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
imagenet_std_tensor = tinygrad.Tensor(imagenet_std)

def normalize(image, mean, std):
    return (image - mean) / std

def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32) / 255.0 
    image_np = np.transpose(image_np, (2, 0, 1))  
    return Tensor(image_np[np.newaxis, ...])  


image1 = image_to_tensor('assets/Chateau1.png')
image2 = image_to_tensor('assets/Chateau2.png')


image1 = (image1 - Tensor(imagenet_mean)) / Tensor(imagenet_std)
image2 = (image2 - Tensor(imagenet_mean)) / Tensor(imagenet_std)

out, mask, target = model(image1, image2)

patchified = model.patchify(image1)
mean = patchified.mean(axis=-1, keepdim=True)
var = patchified.var(axis=-1, keepdim=True)
decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)

decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor
input_image = image1 * imagenet_std_tensor + imagenet_mean_tensor
ref_image = image2 * imagenet_std_tensor + imagenet_mean_tensor
image_masks = model.unpatchify(model.patchify(Tensor.ones_like(ref_image)) * mask[:,:,None])
masked_input_image = ((1 - image_masks) * input_image)



visualization = Tensor.cat(ref_image, masked_input_image, decoded_image, input_image, dim=3)  
B, C, H, W = visualization.shape
visualization = visualization.permute(1, 0, 2, 3).reshape(C, B * H, W)
visualization = visualization.clip(0, 1)
visualization_np = visualization.numpy().transpose(1, 2, 0)  
visualization_np = (visualization_np * 255).astype(np.uint8)  

visualization_image = Image.fromarray(visualization_np)
fname = "demo_output.png"
visualization_image.save(fname)

print(f'Visualization saved in {fname}')