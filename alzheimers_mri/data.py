import numpy as np
from PIL import Image, ImageStat

from .constants import IMAGE_SIZE


def validate_mri(image: Image.Image) -> tuple[bool, str]:
    """Reject unreadable, tiny, blank, or implausibly shaped uploads."""
    if image.width < 64 or image.height < 64:
        return False, "Image is too small; upload a scan at least 64 × 64 pixels."
    ratio = image.width / image.height
    if ratio < 0.45 or ratio > 2.2:
        return False, "Image aspect ratio is unusual for a 2D MRI slice."
    grayscale = image.convert("L")
    stats = ImageStat.Stat(grayscale)
    if stats.stddev[0] < 3:
        return False, "Image appears blank or has too little contrast."
    pixels = np.asarray(grayscale.resize((64, 64)), dtype=np.uint8)
    if np.unique(pixels).size < 8:
        return False, "Image has insufficient intensity variation for analysis."
    return True, "Basic image checks passed. This does not verify that the image is an MRI."


def preprocess_image(image: Image.Image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)
