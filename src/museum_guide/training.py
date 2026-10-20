"""Training tools of the museum guide: train for one epoch, evaluate, and the full training loop.

They follow the functions written in the course (TP05 and TP06), so that every model
of the project is trained and measured in exactly the same way.
"""

import time

import torch


def count_parameters(model):
    """Return the number of weights and biases of a model (its size), as in TP06."""
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, loss_fn, optimizer, device="cpu"):
    """Train the model for one pass over the training data.

    Args:
        model: the network to train.
        loader: DataLoader of the training images.
        loss_fn: the loss, for example nn.CrossEntropyLoss().
        optimizer: the optimiser, for example torch.optim.SGD(...).
        device: "cpu" or "cuda".

    Returns:
        The mean training loss over the epoch.
    """
    model.train()  # training mode: dropout and batch norm behave as during training
    total_loss, n_images = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        # The 5 lines of every PyTorch training loop (TP02, exercise 16)
        optimizer.zero_grad()       # 1. reset the gradients of the previous mini-batch
        logits = model(xb)          # 2. forward pass
        loss = loss_fn(logits, yb)  # 3. loss of this mini-batch
        loss.backward()             # 4. backpropagation: compute the gradients
        optimizer.step()            # 5. update the weights
        total_loss += loss.item() * len(yb)
        n_images += len(yb)
    return total_loss / n_images


def evaluate(model, loader, loss_fn, device="cpu"):
    """Measure the model on a dataset, without training it.

    Args:
        model: the network to measure.
        loader: DataLoader of the images to measure it on.
        loss_fn: the loss.
        device: "cpu" or "cuda".

    Returns:
        (mean loss, accuracy), the accuracy being the fraction of correct predictions.
    """
    model.eval()  # evaluation mode: no dropout, batch norm uses its running statistics
    total_loss, correct, n_images = 0.0, 0, 0
    with torch.no_grad():  # no gradient needed: faster and uses less memory
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            # The predicted class is the one with the highest score (logit)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n_images += len(yb)
    return total_loss / n_images, correct / n_images


def fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device="cpu", verbose=True):
    """Train the model for n_epochs and record its metrics after every epoch, as in TP06.

    The train metrics are measured in evaluation mode on the whole training set after
    each epoch, so that train and validation are measured in the same way.

    Args:
        model, loss_fn, optimizer, device: as in train_one_epoch.
        train_loader, val_loader: DataLoaders of the training and validation images.
        n_epochs: number of passes over the training data.
        verbose: print one line per epoch if True.

    Returns:
        A history: a dictionary of lists with one value per epoch, for the keys
        "train_loss", "train_acc", "val_loss", "val_acc" and "epoch_time" (in seconds).
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    for epoch in range(1, n_epochs + 1):
        # Time the training only, to answer the director's question on CPU cost
        start = time.perf_counter()
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        epoch_time = time.perf_counter() - start

        train_loss, train_acc = evaluate(model, train_loader, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        if verbose:
            print(
                f"epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
                f"| val loss {val_loss:.3f} acc {val_acc:.3f} | {epoch_time:.1f} s"
            )
    return history