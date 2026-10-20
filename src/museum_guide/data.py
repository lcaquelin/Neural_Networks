"""Data pipeline of the museum guide: load, split, standardise and batch the WikiArt subsets.

These functions are shared by all the notebooks, so that every model is trained
and evaluated on exactly the same images.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# This file is src/museum_guide/data.py, so the project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Folder of each subset, as created by scripts/make_wikiart_subset.py
SUBSET_DIRS = {"style": "wikiart_style_all", "genre": "wikiart_genre_all"}

BATCH_SIZE = 64  # images per training mini-batch, as in TP06


def load_as_tensors(dataset, batch_size=256):
    """Read all the images of an ImageFolder dataset into memory, as in TP05.

    Args:
        dataset: an ImageFolder dataset whose transform returns tensors.
        batch_size: number of images read at a time (only affects speed).

    Returns:
        X: float32 tensor of shape (N, 3, H, W), pixel values in [0, 1].
        y: int64 tensor of shape (N,), the class index of each image.
    """
    # Read the images in order (no shuffling), so that X[i] is image number i of the dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images, labels = [], []
    for xb, yb in loader:
        images.append(xb)
        labels.append(yb)
    # Glue all the batches together into one tensor
    return torch.cat(images), torch.cat(labels)


def split_indices(targets, val_size=0.15, test_size=0.15, seed=0):
    """Split the image indices into train / validation / test sets, keeping the class proportions.

    Args:
        targets: the class index of every image.
        val_size: fraction of all the images used for validation.
        test_size: fraction of all the images used for the final test.
        seed: random seed, so that the split is the same at every run.

    Returns:
        Three arrays of image indices: train, validation and test.
    """
    targets = np.array(targets)
    indices = np.arange(len(targets))

    # First set the test images aside...
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=targets, random_state=seed
    )

    # ...then split what is left into train and validation.
    # 15 % of all the images is 15 / 85 = 17.6 % of what is left.
    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, stratify=targets[train_val_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def standardise(X, train_idx):
    """Standardise the images in place, with the statistics of the training images only.

    Each colour channel (red, green, blue) gets its own mean and standard deviation,
    computed on the training images only. They are then applied to all the images,
    so that the validation and test images stay "unseen".

    Args:
        X: float tensor of shape (N, 3, H, W), modified in place.
        train_idx: indices of the training images.

    Returns:
        mean, std: tensors of shape (1, 3, 1, 1), to keep for standardising new images later.
    """
    X_train = X[torch.as_tensor(train_idx)]  # as_tensor: turn the NumPy indices into a tensor
    # Average over the images (dim 0), the rows (dim 2) and the columns (dim 3): one value per channel
    mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    std = X_train.std(dim=(0, 2, 3), keepdim=True)
    # In place (trailing underscore): no second copy of the whole dataset in memory
    X.sub_(mean).div_(std)
    return mean, std


def make_loaders(X, y, train_idx, val_idx, test_idx, batch_size=BATCH_SIZE):
    """Build the train, validation and test DataLoaders of one subset.

    Subset only stores the list of indices, so no image is copied in memory.

    Args:
        X: standardised images, tensor of shape (N, 3, H, W).
        y: class indices, tensor of shape (N,).
        train_idx, val_idx, test_idx: indices of the images of each set.
        batch_size: number of training images per mini-batch.

    Returns:
        train_loader, val_loader, test_loader.
    """
    full_set = TensorDataset(X, y)
    # Shuffle the training images at every epoch, so that the mini-batches always differ
    train_loader = DataLoader(Subset(full_set, train_idx), batch_size=batch_size, shuffle=True)
    # No shuffling for evaluation; bigger batches are faster since no gradient is computed
    val_loader = DataLoader(Subset(full_set, val_idx), batch_size=1000)
    test_loader = DataLoader(Subset(full_set, test_idx), batch_size=1000)
    return train_loader, val_loader, test_loader


def prepare_subset(label, seed=0, batch_size=BATCH_SIZE):
    """Run the whole pipeline for one subset: load, split, standardise and batch.

    Args:
        label: "style" or "genre".
        seed: random seed of the split.
        batch_size: number of training images per mini-batch.

    Returns:
        A dictionary with the class names, the tensors, the indices of each set,
        the standardisation statistics and the three DataLoaders.
    """
    image_set = datasets.ImageFolder(DATA_DIR / SUBSET_DIRS[label], transform=transforms.ToTensor())
    X, y = load_as_tensors(image_set)
    train_idx, val_idx, test_idx = split_indices(y.numpy(), seed=seed)
    mean, std = standardise(X, train_idx)
    train_loader, val_loader, test_loader = make_loaders(
        X, y, train_idx, val_idx, test_idx, batch_size=batch_size
    )
    return {
        "classes": image_set.classes,
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean": mean,
        "std": std,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }