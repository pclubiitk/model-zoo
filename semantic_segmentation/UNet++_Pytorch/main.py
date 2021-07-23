import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import yaml

from dataloader import *
from model import *
from train import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", default=16, type=int)

    # model config
    parser.add_argument("--deep_supervision", default=False, type=str2bool)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--init_features", default=32, type=int)

    # optimizer config
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", "--learning_rate", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    # scheduler config
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--early_stopping", default=-1, type=int)

    # run config
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", default=2, type=int)

    config = parser.parse_args()

    return config


config = vars(parse_args())

os.makedirs("models/", exist_ok=True)
with open("models/config.yml", "w") as f:
    yaml.dump(config, f)


model = UNetPP(
    config["input_channels"], config["num_classes"], config["deep_supervision"], config["init_features"]
)
model = model.to(config["device"])
params = filter(lambda p: p.requires_grad, model.parameters())

criterion = BCEDiceLoss

if config["optimizer"] == "Adam":
    optimizer = optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
else:
    optimizer = optim.SGD(
        params, lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"]
    )

scheduler = CosineAnnealingLR(
    optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
)

train_dl, test_dl = get_loader(config["batch_size"], config["num_workers"])

log = train(
    config, train_dl, test_dl, model, optimizer, scheduler, criterion, metric=iou
)

# analysis
plot_log(log)
