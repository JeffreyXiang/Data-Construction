from PIL import Image
import numpy as np
import torch
import cv2
from models import ivid
try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    print("\033[93mWarning: StableDiffusionInpaintPipeline not found. Please install diffusers.\033[0m")


def load_sdv2_inpaint_model(model_path="stabilityai/stable-diffusion-2-inpainting", device='cuda'):
    sdv2inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    return sdv2inpaint_model


def load_ivid_inpaint_model(model_name='sdipdogs_depth_aware_inpaint'):
    ivid_inpaint_model = ivid.load(model_name)
    return ivid_inpaint_model


@torch.no_grad()
def sdv2_inpaint(sdv2inpaint_model, image: torch.Tensor, mask: torch.Tensor, prompt: str = ""):
    """
    Inpaint image using StableDiffusion.
    Args:
        image: (N, 3, H, W) tensor of images.
        mask: (N, H, W) tensor of masks.
        prompt: list of prompts.
    """
    image_np = np.clip(image.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
    for i in range(len(mask_np)):
        mask_np[i] = cv2.dilate(mask_np[i], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    image_pil = [Image.fromarray(image_np[i]) for i in range(image_np.shape[0])]
    mask_pil = [Image.fromarray(mask_np[i]) for i in range(mask_np.shape[0])]
    if isinstance(prompt, str):
        prompt = [prompt] * len(image_pil)
    out = sdv2inpaint_model(prompt=prompt, image=image_pil, mask_image=mask_pil).images
    out = [np.array(img) for img in out]
    out = np.stack(out, axis=0)
    # Postprocess with poisson blending
    for i in range(out.shape[0]):
        # find bbox of mask
        mask_bbox = cv2.boundingRect(255 - mask_np[i])
        center = (mask_bbox[0] + mask_bbox[2] // 2, mask_bbox[1] + mask_bbox[3] // 2)
        out[i] = cv2.seamlessClone(
            image_np[i], out[i],
            255 - mask_np[i],
            center, cv2.NORMAL_CLONE
        )
    out = torch.tensor(out).permute(0, 3, 1, 2).float().to(image.device) / 255.0
    return out


@torch.no_grad()
def ivid_inpaint(ivid_inpaint_model, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
    return ivid.inpaint(ivid_inpaint_model, image, depth, mask)
