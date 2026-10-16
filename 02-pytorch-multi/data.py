"""In-memory synthetic images; no network, downloads, or worker subprocesses."""
import torch
from torch.profiler import record_function
from torch.utils.data import DataLoader, Dataset


class PatternDataset(Dataset):
    def __init__(self, count, image_size, seed=42):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, image_size, image_size, generator=generator)
        self.labels = torch.arange(count) % 4
        # Four labels have different channel intensities, so labels are not random noise.
        for label in range(4):
            mask = self.labels == label
            self.images[mask, label % 3] += 0.35 * (1 + label // 3)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def make_loader(batch_size, image_size):
    dataset = PatternDataset(batch_size * 4, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def prepare_batch(batch, device, training):
    with record_function("preprocess"):
        images, labels = batch
        with record_function("augmentation"):
            if training:
                images = torch.flip(images, dims=[-1])
        with record_function("normalize"):
            images = (images - 0.5) / 0.5
    with record_function("device_transfer"):
        return images.to(device), labels.to(device)
