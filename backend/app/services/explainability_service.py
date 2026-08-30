import gc
from pathlib import Path

import cv2
import numpy as np
import torch


def generate_gradcam(
    model,
    image_tensor,
    feature_tensor,
    original_image,
    output_dir,
    output_filename=None,
):
    """Generate a quality-output Grad-CAM and release all graph state on exit."""
    model.eval()
    target_layer = model.backbone.layer4[-1].conv2
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, hook_input, output):
        # Detaching in the hook prevents it retaining the full forward graph.
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = backward_handle = None
    quality = defect_output = quality_score = None
    activation = gradient = weights = cam_tensor = None
    cam = heatmap = overlay = None
    try:
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        # Grad-CAM is the only production operation that enables autograd.
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            # Parameters have gradients disabled; differentiating through this input
            # still captures the target layer gradient without .grad model buffers.
            image_tensor.requires_grad_(True)
            quality, defect_output = model(image_tensor, feature_tensor)
            quality_score = quality[0]
            quality_score.backward()

        if not activations:
            raise RuntimeError("Grad-CAM activations were not captured.")
        if not gradients:
            raise RuntimeError("Grad-CAM gradients were not captured.")

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam_tensor = torch.relu((weights * activation).sum(dim=1)).squeeze(0)
        cam = cam_tensor.cpu().numpy()

        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        height, width = original_image.shape[:2]
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image, 0.55, heatmap, 0.45, 0)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_filename or f"gradcam_{np.random.randint(100000, 999999)}.jpg"
        if not cv2.imwrite(str(output_dir / filename), overlay):
            raise RuntimeError("Failed to save Grad-CAM image.")
        return filename
    finally:
        # Hooks and graph-bearing outputs must never survive a failed request.
        if forward_handle is not None:
            forward_handle.remove()
        if backward_handle is not None:
            backward_handle.remove()
        activations.clear()
        gradients.clear()
        model.zero_grad(set_to_none=True)
        del quality, defect_output, quality_score, activation, gradient, weights, cam_tensor
        del cam, heatmap, overlay
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
