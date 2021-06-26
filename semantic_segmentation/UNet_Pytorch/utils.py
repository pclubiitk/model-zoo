import torch
import matplotlib.pyplot as plt


# loss function
def BCEDiceLoss(outputs, labels, smooth=1e-5):
    bce = torch.nn.BCEWithLogitsLoss()(outputs, labels)

    outputs = torch.sigmoid(outputs)
    batch_size = outputs.shape[0]
    outputs = outputs.view(batch_size, -1)
    labels = labels.view(batch_size, -1)
    intersection = outputs * labels

    dice = (2.0 * intersection.sum(1) + smooth) / (
        outputs.sum(1) + labels.sum(1) + smooth
    )
    dice = 1 - dice.sum() / batch_size

    return bce / 2 + dice


# metrics
def dice_coef(outputs, labels, smooth=1e-5):
    outputs = torch.sigmoid(outputs).view(-1).data.cpu().numpy()
    outputs = outputs > 0.5
    labels = labels.view(-1).data.cpu().numpy()
    intersection = (outputs * labels).sum()

    dice = (2.0 * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
    return dice


def iou(outputs, labels, smooth=1e-5):
    outputs = torch.sigmoid(outputs).data.cpu().numpy()
    outputs = outputs > 0.5
    labels = labels.data.cpu().numpy()
    labels = labels > 0.5

    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    return (intersection + smooth) / (union + smooth)


# plots
def plot_log(log):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(log["epoch"], log["train_loss"], label="Training Loss")
    plt.plot(log["epoch"], log["val_loss"], label="Validation Loss")
    plt.axis([0, len(log["epoch"]), 0, 2])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log["epoch"], log["train_metric"], label="Training IoU")
    plt.plot(log["epoch"], log["val_metric"], label="Validation IoU")
    plt.axis([0, len(log["epoch"]), 0, 1])
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()
    plt.show()
