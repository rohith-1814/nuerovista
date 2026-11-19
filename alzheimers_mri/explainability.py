import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .constants import IMAGE_SIZE
from .data import preprocess_image


def _target_layer(model, model_name: str):
    return model.layer4[-1] if model_name == "resnet50" else model.features[-1]


def grad_cam(model, model_name: str, input_tensor: torch.Tensor, class_index: int):
    activations = []
    gradients = []
    layer = _target_layer(model, model_name)
    forward_hook = layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    backward_hook = layer.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0])
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[0, class_index].backward()
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap.detach().cpu().numpy()
        maximum = float(heatmap.max())
        return heatmap / maximum if maximum > 0 else heatmap
    finally:
        forward_hook.remove()
        backward_hook.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    import matplotlib

    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    colors = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colors).resize(base.size)
    return Image.blend(base, heat, alpha=0.4)


def shap_superpixel_values(model, image: Image.Image, device: torch.device):
    import shap

    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(resized, dtype=np.uint8)
    segments = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    grid = 4
    for row in range(grid):
        for column in range(grid):
            segments[
                row * IMAGE_SIZE // grid : (row + 1) * IMAGE_SIZE // grid,
                column * IMAGE_SIZE // grid : (column + 1) * IMAGE_SIZE // grid,
            ] = row * grid + column

    def predict(mask_batch):
        masked_images = []
        for mask in np.asarray(mask_batch):
            masked = pixels.copy()
            for segment_id, visible in enumerate(mask):
                if not visible:
                    masked[segments == segment_id] = 0
            masked_images.append(preprocess_image(Image.fromarray(masked)))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(masked_images), 8):
                tensor = torch.cat(masked_images[start : start + 8]).to(device)
                outputs.extend(torch.softmax(model(tensor), dim=1).cpu().numpy())
        return np.asarray(outputs)

    explainer = shap.KernelExplainer(predict, np.zeros((1, grid * grid)))
    explanation = explainer.shap_values(
        np.ones((1, grid * grid)), nsamples=96, silent=True
    )
    probabilities = predict(np.ones((1, grid * grid)))[0]
    predicted_class = int(np.argmax(probabilities))
    if isinstance(explanation, list):
        values = np.asarray(explanation[predicted_class])[0]
    else:
        values_array = np.asarray(explanation)
        values = (
            values_array[0, :, predicted_class]
            if values_array.ndim == 3
            else values_array.reshape(-1)
        )
    grid_values = values.reshape(grid, grid)
    heatmap = np.repeat(np.repeat(grid_values, IMAGE_SIZE // grid, axis=0), IMAGE_SIZE // grid, axis=1)
    maximum = float(np.max(np.abs(heatmap)))
    return heatmap / maximum if maximum > 0 else heatmap
