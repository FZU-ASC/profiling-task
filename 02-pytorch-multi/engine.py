"""Training and evaluation have separate nested profiling ranges."""
import torch
from torch import nn
from torch.profiler import record_function

from data import prepare_batch


def train_step(model, optimizer, batch, device):
    model.train()
    with record_function("train_step"):
        images, labels = prepare_batch(batch, device, training=True)
        with record_function("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with record_function("forward"):
            logits = model(images)
        with record_function("loss"):
            loss = nn.functional.cross_entropy(logits, labels)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
    return loss.detach()


def evaluate(model, batch, device):
    model.eval()
    with record_function("evaluation"), torch.inference_mode():
        images, labels = prepare_batch(batch, device, training=False)
        with record_function("eval_forward"):
            logits = model(images)
        with record_function("metrics"):
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return accuracy
