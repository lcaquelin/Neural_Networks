"""Predict the most likely styles bis and genres of a photo with the two trained ResNet18.

The models are the ResNet18 of `06_01_resnet_style` (27 styles) and `06_02_resnet_genre` (11 genres),
saved by the last cell of each notebook in `data/models/`. For the styles, the 27 style probabilities
are summed by style bis (module `museum_guide.style_bis`): no retraining is needed.
"""

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.models import resnet18
from torchvision.transforms.functional import to_tensor

from museum_guide.data import PROJECT_ROOT
from museum_guide.style_bis import STYLE_BIS_NAMES, style_bis_probabilities

# Weights saved by the last cell of 06_01_resnet_style and 06_02_resnet_genre (data/ is ignored by Git)
STYLE_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "resnet18_style_app.pt"
GENRE_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "resnet18_genre_app.pt"

# The ResNet18 was pretrained on ImageNet: its inputs are normalised with the ImageNet statistics (course S07)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(checkpoint):
    """Load a trained ResNet18 on the CPU.

    Args:
        checkpoint: file saved by the last cell of 06_01_resnet_style or 06_02_resnet_genre.

    Returns:
        (model, class_names, img_size): the network in evaluation mode, the class names in the
        order of its outputs, and the side of the images it was trained on, in pixels.
    """
    saved = torch.load(checkpoint, map_location="cpu")
    model = resnet18(weights=None)  # the architecture only: the trained weights are loaded just below
    model.fc = nn.Linear(model.fc.in_features, len(saved["class_names"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["class_names"], saved["img_size"]


def prepare_photo(image, img_size):
    """Prepare a photo exactly like the training images, then like the ImageNet pre-training.

    Args:
        image: a PIL image (any size, any colour mode).
        img_size: side of the training images, in pixels.

    Returns:
        A tensor of shape (1, 3, img_size, img_size).
    """
    image = ImageOps.exif_transpose(image)  # phone photos: apply the orientation stored by the camera
    image = image.convert("RGB")
    # Same resizing as the course script: shorter side to img_size, then a centre crop
    image = ImageOps.fit(image, (img_size, img_size), method=Image.Resampling.BICUBIC)
    x = to_tensor(image).unsqueeze(0)  # pixels in [0, 1], shape (1, 3, H, W)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def predict_probabilities(model, image, img_size):
    """Return the probability of each class for one photo, as a tensor of shape (1, n_classes)."""
    x = prepare_photo(image, img_size)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)


def predict_top_style_bis(model, class_names, image, img_size, k=3):
    """Return the names of the k most likely styles bis of a photo, the most likely first.

    Args:
        model, class_names, img_size: as returned by load_model for the style model.
        image: a PIL image of a painting.
        k: number of styles bis returned.
    """
    style_probs = predict_probabilities(model, image, img_size)  # one probability per style, shape (1, 27)
    bis_probs = style_bis_probabilities(style_probs, class_names)[0]  # one per style bis, shape (18,)
    top_idx = torch.topk(bis_probs, k).indices
    return [STYLE_BIS_NAMES[i] for i in top_idx.tolist()]


def predict_top_genres(model, class_names, image, img_size, k=3):
    """Return the names of the k most likely genres of a photo, the most likely first.

    Args:
        model, class_names, img_size: as returned by load_model for the genre model.
        image: a PIL image of a painting.
        k: number of genres returned.
    """
    genre_probs = predict_probabilities(model, image, img_size)[0]  # one probability per genre, shape (11,)
    top_idx = torch.topk(genre_probs, k).indices
    return [class_names[i] for i in top_idx.tolist()]
