import torch
import numpy as np
from collections import OrderedDict


def train_step(model, criterion, train_dl, optimizer, deep_supervision, device, metric):
    losses, nums, metrics = [], [], []

    for xb, yb_ in train_dl:

        xb = (xb.to(device)).float()
        yb_ = (yb_.to(device)).float()
        yb = (yb_ > 0) * 1.0

        # compute predictions
        if deep_supervision:
            outputs = model(xb)
            loss = 0
            for output in outputs:
                loss += criterion(output, yb) / len(outputs)
            metric_score = metric(outputs[-1], yb)
        else:
            output = model(xb)
            loss = criterion(output, yb)
            metric_score = metric(output, yb)

        losses.append(loss)
        nums.append(xb.shape[0])
        metrics.append(metric_score)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total = np.sum(nums)
    avg_loss = np.sum(np.multiply(losses, nums)) / total
    avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, avg_metric


def validation_step(model, criterion, dataloader, deep_supervision, device, metric):
    with torch.no_grad():
        losses, nums, metrics = [], [], []

        for xb, yb_ in dataloader:

            xb = (xb.to(device)).float()
            yb_ = (yb_.to(device)).float()
            yb = (yb_ > 0) * 1.0

            if deep_supervision:
                outputs = model(xb)
                loss = 0
                for output in outputs:
                    loss += criterion(output, yb) / len(outputs)
                metric_score = metric(outputs[-1], yb)
            else:
                output = model(xb)
                loss = criterion(output, yb)
                metric_score = metric(output, yb)

            losses.append(loss)
            nums.append(xb.shape[0])
            metrics.append(metric_score)

        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, avg_metric


def train(config, train_dl, valid_dl, model, optimizer, scheduler, criterion, metric):
    best_iou = 0
    trigger = 0
    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("train_loss", []),
            ("train_metric", []),
            ("val_loss", []),
            ("val_metric", []),
        ]
    )

    for epoch in range(config["epochs"]):

        # training step
        model.train()
        train_loss, train_metric = train_step(
            model,
            criterion,
            train_dl,
            optimizer,
            config["deep_supervision"],
            config["device"],
            metric,
        )

        # evaluation step
        model.eval()
        val_loss, val_metric = validation_step(
            model,
            criterion,
            valid_dl,
            config["deep_supervision"],
            config["device"],
            metric,
        )

        scheduler.step()

        print(
            "Epoch [{}/{}, train_loss: {:.4f}, train_{}: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}".format(
                epoch + 1,
                config["epochs"],
                train_loss,
                metric.__name__,
                train_metric,
                val_loss,
                metric.__name__,
                val_metric,
            )
        )

        log["epoch"].append(epoch)
        log["lr"].append(config["lr"])
        log["train_loss"].append(train_loss)
        log["train_metric"].append(train_metric)
        log["val_loss"].append(val_loss)
        log["val_metric"].append(val_metric)

        trigger += 1

        if val_metric > best_iou:
            torch.save(model.state_dict(), "models/model_ep-{}.pth".format(epoch))
            best_iou = val_metric
            print("=> Saved best model")
            trigger = 0

        # early stopping
        if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
            print("=> Early stopping")
            break

        if config["device"] == "cuda":
            torch.cuda.empty_cache()

    return log
