import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict 

from .gaussian_diffusion import GaussianDiffusion


class InpaintCFG(GaussianDiffusion):
    """
    Image inpainting with classifier-free guidance.
    
    Args:
        p_uncond: probability of drop the class label.
        p_uncond_img: probability of drop the inpaint image cond.
    """
    def __init__(self, backbone, *, p_uncond=0.1, p_uncond_img=0.0, **kwargs):
        super().__init__(backbone, **kwargs)
        self.p_uncond = p_uncond
        self.p_uncond_img = p_uncond_img

    def make_cond_inputs(self, x, y, mask, **kwargs):
        """
        Make inputs for the conditional model.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            y: The [N x C x ...] tensor of ground truth partially visible images.
            mask: The [N x 1 x ...] tensor of visibility masks.
        """
        y_rgb = y[:, :3]
        y_depth = y[:, 3:]
        in_list = [x]

        if 'mask_rgb' in kwargs:
            mask_rgb = kwargs['mask_rgb']
            in_list.append(mask_rgb)
        else:
            mask_rgb = mask
        
        y_rgb = y_rgb * mask_rgb + torch.randn_like(y_rgb) * (1 - mask_rgb)
        in_list.append(y_rgb)
        y_depth = y_depth * mask + torch.randn_like(y_depth) * (1 - mask)
        in_list.append(y_depth)
        in_list.append(mask)
        
        return torch.cat(in_list, dim=1)

    def make_uncond_inputs(self, x):
        """
        Make inputs for the unconditional model.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
        """
        return torch.cat([x, torch.randn_like(x), torch.zeros_like(x[:, :1])], dim=1)

    @torch.no_grad()
    def model_inference(self, x, t, y, mask, classes=None, strength=3.0, **kwargs):
        """
        Do inference with the backbone, returning the predicted noise.

        Args:
            x: The [N x C x ...] tensor of noisy inputs at time t.
            t: The [N] tensor of diffusion steps (minus 1). Here, 0 means one step.
            y: The [N x C x ...] tensor of ground truth partially visible images.
            mask: The [N x 1 x ...] tensor of visibility masks.
            classes: The [N] tensor of class labels.
            strength: The strength of the classifier-free guidance.
        
        Returns:
            The [N x C x ...] tensor of predicted noise.
        """
        cond_inputs = self.make_cond_inputs(x, y, mask, **kwargs)
        # uncond_inputs = self.make_uncond_inputs(x)
        if classes is None:
            return self.backbone(cond_inputs, t, None)
        return (
            (1 + strength) * self.backbone(cond_inputs, t, classes)
            - (strength * self.backbone(cond_inputs, t, None) if strength > 0 else 0)
        )

    def training_losses(self, x_0, y, mask, classes=None, **kwargs):
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            y: The [N x C x ...] tensor of ground truth partially visible images.
            mask: The [N x 1 x ...] tensor of visibility masks.
            classes: The [N] tensor of class labels.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        x_t = self.diffuse(x_0, t, noise=noise)

        # randomly drop the class label
        if classes is not None and self.p_uncond > 0:
            classes = torch.where(
                torch.rand_like(classes.float()) < self.p_uncond,
                -torch.ones_like(classes),
                classes,
            )

        # randomly drop the inpaint image cond
        if self.p_uncond_img > 0:
            x_t = torch.where(
                torch.rand(x_t.shape[0], 1, 1, 1, device=x_t.device) < self.p_uncond_img,
                self.make_uncond_inputs(x_t),
                self.make_cond_inputs(x_t, y, mask),
            )
        else:
            x_t = self.make_cond_inputs(x_t, y, mask, **kwargs)

        pred_eps = self.backbone(x_t, t, classes)
        assert pred_eps.shape == noise.shape == x_0.shape

        terms = edict()
        terms["mse"] = F.mse_loss(pred_eps, noise)
        terms["loss"] = 1 * terms["mse"]

        return terms
