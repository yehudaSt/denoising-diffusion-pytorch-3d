
from glob import glob
import os

import matplotlib.pyplot as plt
import torch
from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.repaint import GaussianDiffusion as RePaint
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import pil_to_tensor


# Taken from: https://github.com/lucidrains/denoising-diffusion-pytorch/issues/317
class FFHQDataset(VisionDataset):
    def __init__(self, root: str):
        super().__init__(root)

        self.fpaths = sorted(glob(root + "/*.png", recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")
        # todo: keep aspect ratio:
        img = img.resize((128, 128))
        # todo: use depth instead of alpha and normalize it to [0, 1]
        gray_img = img.convert("L")
        img.putalpha(gray_img)
        # normalize to [0, 1] range
        img = pil_to_tensor(img) / 255.0
        return img


def create_center_square_mask(image_size: int, mask_size: int):
    assert image_size >= mask_size, "Mask size should be smaller or equal to image size"

    mask = torch.zeros((image_size, image_size))
    start = (image_size - mask_size) // 2
    end = start + mask_size
    mask[start:end, start:end] = 1

    return (mask - 1) * -1

def create_middle_column_mask(image_size, mask_size: int):
    """Create a mask that is a center column down the image"""
    mask = torch.zeros((image_size, image_size))
    start = (image_size - mask_size) // 2
    end = start + mask_size
    mask[:, start:end] = 1
    return (mask - 1) * -1


def plot_results(target, masked_gt, mask, inpainted, dir):
    """Plot results.
    Args:
        target: full target
        masked_gt: target with mask applied
        mask: mask tensor
        inpainted: inpainted tensor
    """
    batch_size = target.size(0)

    fig, axs = plt.subplots(batch_size, 6, figsize=(30, 5 * batch_size))

    for i in range(batch_size):
        target_np = target[i].numpy().transpose(1, 2, 0)
        masked_gt_np = masked_gt[i].numpy().transpose(1, 2, 0)
        mask_np = mask[i].numpy().transpose(1, 2, 0)
        inpainted_np = inpainted[i].numpy().transpose(1, 2, 0)

        # TODO: remove: fix alpha channel:

        axs[i, 0].imshow(target_np[:, :, 0:3])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target_np[:, :, 3], cmap="gray")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(masked_gt_np[:, :, 3], cmap="gray")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(mask_np, cmap="gray")
        axs[i, 3].axis("off")

        axs[i, 4].imshow(inpainted_np[:, :, 3], cmap="gray") # plot gray image (i.e. depth)
        axs[i, 4].axis("off")

        axs[i, 5].imshow(inpainted_np[:, :, 0:3]) # plot gray image (i.e. depth)
        axs[i, 5].axis("off")

    axs[0, 0].set_title("Original Image", fontsize=40)
    axs[0, 1].set_title("Original Depth", fontsize=40)
    axs[0, 2].set_title("Masked Input", fontsize=40)
    axs[0, 3].set_title("Mask", fontsize=40)
    axs[0, 4].set_title("Inpainted Image", fontsize=40)
    axs[0, 5].set_title("Inpainted Color", fontsize=40)

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.tight_layout()

    fig.savefig(os.path.join(dir, "inpainted_image.png"))

def main():

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=4) # dim_mults=(1, 2, 4, 8, 16, 32), flash_attn=True)

    diffusion = RePaint(
        model, image_size=128, timesteps=1000, sampling_timesteps=500
    )

    # pretrained model on FFHQ
    ckpt_path = "/home/onesh/repos/denoising-diffusion-pytorch/denoising_diffusion_pytorch/training/results/model-final_1000_3d.pt"
    diffusion.load_state_dict(torch.load(ckpt_path)["model"])


    # batch from dataloader
    ds = FFHQDataset(
        root="/home/onesh/data/dividers"
    )
    dl = DataLoader(ds, batch_size=5, shuffle=True)
    imgs = next(iter(dl))

    # +1 values stand for areas to keep and 0 for areas to be inpainted
    image_size = imgs.shape[-1]
    # TODO: keep aspect ratio:
    mask_size = image_size // 3
    masks = (
        create_center_square_mask(image_size, mask_size)
        .repeat(imgs.shape[0], 1, 1)
        .unsqueeze(1)
    )

    # Apply the mask to the image
    # Apply the mask only to the 4th channel since that is where the depth corruption is
    masked_imgs = imgs.detach().clone()
    for i in range(imgs.shape[0]):
        masked_imgs[i, 3] = imgs[i, 3] * masks[i]
    # masked_imgs = imgs * masks

    # move to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion = diffusion.to(device)
    masked_imgs = masked_imgs.to(device)
    masks = masks.to(device)

    # generate inpainting
    inpainted = diffusion.sample(gt=masked_imgs, mask=masks)
    # min-max normalization for plotting
    inpainted = (inpainted - inpainted.min()) / (inpainted.max() - inpainted.min())


    plot_results(imgs.to("cpu"), masked_imgs.to("cpu"), masks.to("cpu"), inpainted.to("cpu"), dir=os.path.dirname(ckpt_path))

if __name__ == "__main__":
    main()