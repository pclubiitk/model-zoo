import argparse
import torch
import os

from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from dataloader import load_dataset
from train import train_st, stylize, train_sr, enhance
from models import SuperResolutionNet, StyleTransferNet, LossNet_VGG16


def main():
    config = parse_args()

    lossnet = (LossNet_VGG16()).to(config.device)
    lossnet.eval()

    os.makedirs(config.sample_path, exist_ok=True)
    os.makedirs(config.model_path, exist_ok=True)

    if config.mode == "style_transfer":
        stnet = StyleTransferNet(
            config.in_channels, config.init_features, config.num_residuals
        )
        stnet.to(config.device)

        train_dl = load_dataset(
            config.batch_size, config.image_size, config.num_workers, config.split
        )

        style_img = Image.open(config.style_image).convert("RGB")
        content_img = Image.open(config.content_image).convert("RGB")
        with torch.no_grad():
            style_transform = transforms.Compose(
                [
                    transforms.Resize(config.image_size),
                    transforms.CenterCrop(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            style_img = (
                (style_transform(style_img))
                .repeat(config.batch_size, 1, 1, 1)
                .to(config.device)
            )
            content_img = (style_transform(content_img)).unsqueeze(0).to(config.device)

        optimizer = torch.optim.Adam(stnet.parameters(), lr=config.lr)
        torch.set_default_tensor_type(torch.FloatTensor)

        train_st(config, train_dl, style_img, content_img, stnet, lossnet, optimizer)
        stylize(config, train_dl, stnet, config.samples)

    elif config.mode == "super_resolution":
        srnet = SuperResolutionNet(
            config.in_channels,
            config.init_features,
            config.num_residuals,
            config.upscale_factor,
        )
        srnet.to(config.device)

        train_dl = load_dataset(
            config.batch_size,
            config.image_size,
            config.num_workers,
            config.split,
            config.upscale_factor,
        )

        c_x = Image.open(config.content_image).convert("RGB")
        c_target = Image.open(config.content_image).convert("RGB")
        with torch.no_grad():
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        config.image_size // config.upscale_factor,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(config.image_size // config.upscale_factor),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            c_x = (transform(c_x)).unsqueeze(0).to(config.device)

            c_target = transforms.Resize(config.image_size)(c_target)
            c_target = transforms.CenterCrop(config.image_size)(c_target)

        Image.Image.save(
            c_target,
            config.sample_path + "sample_target.jpg",
        )

        optimizer = torch.optim.Adam(srnet.parameters(), lr=config.lr)
        torch.set_default_tensor_type(torch.FloatTensor)

        train_sr(config, train_dl, srnet, lossnet, optimizer, c_x)
        enhance(config, train_dl, srnet, samples=5)

    else:
        print(f"Unknown mode {config.mode}")
        print("Available modes: style_transfer, super_resolution")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="style_transfer",
        choices=["style_transfer", "super_resolution"],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--init_features", type=int, default=32)
    parser.add_argument("--num_residuals", type=int, default=5)
    parser.add_argument("--upscale_factor", type=int, default=4, choices=[4, 8])

    parser.add_argument("--style_image", type=str, default="/assets/style/the_muse.jpg")
    parser.add_argument("--content_image", type=str, default="/assets/content/taj_mahal.jpg")
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument(
        "--style_weights", nargs="+", type=float, default=[5e4, 3e4, 5e4, 5e4]
    )
    parser.add_argument("--reg_weight", type=float, default=1e-7)

    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_step", type=int, default=200)
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample_path", type=str, default="/samples/")
    parser.add_argument("--model_path", type=str, default="/models/")

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    main()
