# [num_batched,n, 4], [num_batch ,n, 4]


import time
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_data: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    minibatch=False,
):
    model.train()
    total_loss = 0
    start_time = time.time()
    interval = 100
    if minibatch:
        train_data = [next(iter(train_data))]

    total_batches = len(train_data) - 1
    for batch_i, batch in enumerate(train_data):
        output = model(
            batch["en_ids"], batch["de_ids"], batch["en_att"], batch["de_att"]
        )
        loss = criterion(output, batch[)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_i % interval == 0 and batch_i > 0:
            create_report(
                writer, total_loss / interval, batch_i, total_batches, start_time, epoch
            )
            total_loss = 0
            start_time = time.time()


def create_report(
    writer: SummaryWriter,
    loss: float,
    batch_i: int,
    total_batches: int,
    start_time: float,
    epoch: int,
):
    index = batch_i + epoch * total_batches
    tm = time.time() - start_time
    writer.add_scalar("Loss/train", loss, index)
    writer.add_scalar("Time/train", tm, index)
    print(
        "Progress/train", f"{epoch}: {batch_i}/{total_batches} Loss: {loss}, Time: {tm}"
    )


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    test_data: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    n_labels: int,
    minibatch=False,
):
    # If minibatch is True test_data must not be shuffle in order for this to work
    model.eval()
    total_loss = 0
    predictions = []
    correct_labels = []
    if minibatch:
        test_data = [next(iter(test_data))]

    with torch.no_grad():
        for sentences, label in test_data:

            output = model(*sentences)
            predictions.append(output.argmax(dim=1))
            correct_labels.append(label)
            total_loss += criterion(output, label).item()

    predictions = torch.cat(predictions, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)
    acc = Accuracy(num_classes=n_labels, average="macro").to(device)(
        predictions, correct_labels
    )
    prec = Precision(num_classes=n_labels, average="macro").to(device)(
        predictions, correct_labels
    )
    recall = Recall(num_classes=n_labels, average="macro").to(device)(
        predictions, correct_labels
    )
    f1 = F1Score(num_classes=n_labels, average="macro").to(device)(
        predictions, correct_labels
    )
    loss_avg = total_loss / len(test_data)
    print(f"{epoch} Acc: {acc} Loss: {loss_avg}")
    writer.add_scalar("Loss/dev", loss_avg, epoch)
    writer.add_scalar("Acc/dev", acc, epoch)
    writer.add_scalar("Prec/dev", prec, epoch)
    writer.add_scalar("Recall/dev", recall, epoch)
    writer.add_scalar("F1/dev", f1, epoch)
    return total_loss / len(test_data)
