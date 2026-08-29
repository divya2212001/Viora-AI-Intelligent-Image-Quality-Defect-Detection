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

    """
    Generate Grad-CAM for the quality prediction
    of the Hybrid CNN + CV model.
    """

    model.eval()
    target_layer = (
        model.backbone.layer4[-1].conv2
    )

    activations = []
    gradients = []


    def forward_hook(
        module,
        input,
        output,
    ):

        activations.append(
            output
        )

    def backward_hook(
        module,
        grad_input,
        grad_output,
    ):
        gradients.append(
            grad_output[0]
        )


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        model.zero_grad(
            set_to_none=True
        )

        quality, _ = model(
            image_tensor,
            feature_tensor,
        )

        quality_score = (
            quality[0]
        )

        quality_score.backward()

        if not activations:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )
        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activation = (
            activations[0]
            .detach()
            .cpu()
            .numpy()[0]
        )

        gradient = (
            gradients[0]
            .detach()
            .cpu()
            .numpy()[0]
        )

        weights = np.mean(
            gradient,
            axis=(1, 2),
        )

        cam = np.zeros(
            activation.shape[1:],
            dtype=np.float32,
        )

        for index, weight in enumerate(
            weights
        ):

            cam += (
                weight
                * activation[index]
            )

        cam = np.maximum(
            cam,
            0,
        )

        maximum = cam.max()
        if maximum > 0:

            cam = (
                cam
                / maximum
            )
        height, width = (
            original_image.shape[:2]
        )
        cam = cv2.resize(
            cam,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap = np.uint8(
            cam * 255
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            original_image,
            0.55,
            heatmap,
            0.45,
            0,
        )
        output_dir = Path(
            output_dir
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        filename = output_filename or (
            "gradcam_"
            + str(np.random.randint(100000, 999999))
            + ".jpg"
        )


        output_path = (
            output_dir
            / filename
        )


        success = cv2.imwrite(
            str(output_path),
            overlay,
        )


        if not success:

            raise RuntimeError(
                "Failed to save Grad-CAM image."
            )


        return filename


    finally:

        forward_handle.remove()
        backward_handle.remove()
