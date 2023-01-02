from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torchmetrics import BLEUScore
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import time
from transfomer import Transformer


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


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    criterion: nn.Module,
    train_data: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    minibatch=False,
    device=torch.device("cpu"),
):
    model.train()
    total_loss = 0
    start_time = time.time()
    if minibatch:
        train_data = [next(iter(train_data))]
    interval = len(train_data) // 15

    total_batches = len(train_data) - 1
    for batch_i, batch in enumerate(train_data):
        # Remove last token from target as it is not required
        target_ids = batch["de_ids"][:, :-1]
        target_att = batch["de_att"][:, :, :, :-1]
        source_ids = batch["en_ids"]
        source_mask = batch["en_att"] == 1

        target_mask = Transformer.make_trg_mask(target_att == 1)
        output = model(source_ids, target_ids, source_mask, target_mask)

        target_correct = batch["de_ids"][:, 1:].to(device)
        loss = criterion(output.transpose(1, 2), target_correct)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_i % interval == 0 and batch_i > 0 or minibatch:
            create_report(
                writer, total_loss / interval, batch_i, total_batches, start_time, epoch
            )
            total_loss = 0
            start_time = time.time()


def strip_start_end(tokens, end_token):
    tokens = tokens[1:]
    try:
        end = tokens.index(end_token)
        return tokens[:end]
    except ValueError:
        pass

    return tokens


def evaluate(
    model: nn.Module,
    test_data: DataLoader,
    writer: SummaryWriter,
    de_tokenizer: Tokenizer,
):
    start_token, end_token = de_tokenizer.token_to_id(
        "[START]"
    ), de_tokenizer.token_to_id("[END]")
    model.eval()
    total_bleu = 0
    bleu_score = BLEUScore()
    total_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(test_data):
            total_samples += len(batch["de_sent"])
            # Remove last token from target as it is not required
            source_ids = batch["en_ids"]
            source_mask = batch["en_att"] == 1

            target_sent = batch["de_sent"]
            predicted = model.predict(source_ids, source_mask, start_token, end_token)
            # Convertes predicted to list of sentences and then strips the start and end tokens
            without_start_end = [
                strip_start_end(p, end_token=end_token) for p in predicted.tolist()
            ]
            truth = [[str(x)] for x in target_sent]
            decoded_predictions = de_tokenizer.decode_batch(without_start_end)
            bleu = bleu_score(decoded_predictions, truth)

            total_bleu += bleu

    result = total_bleu / total_samples
    writer.add_scalar("BLEU", result)
    return result
