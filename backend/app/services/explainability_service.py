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
):

    """
    Generate Grad-CAM for the quality prediction
    of the Hybrid CNN + CV model.
    """

    model.eval()


    # =====================================================
    # TARGET CNN LAYER
    # =====================================================

    target_layer = (
        model.backbone.layer4[-1].conv2
    )


    activations = []
    gradients = []


    # =====================================================
    # FORWARD HOOK
    # =====================================================

    def forward_hook(
        module,
        input,
        output,
    ):

        activations.append(
            output
        )


    # =====================================================
    # BACKWARD HOOK
    # =====================================================

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

        # =================================================
        # ENABLE GRADIENTS
        # =================================================

        model.zero_grad(
            set_to_none=True
        )


        # =================================================
        # FORWARD
        # =================================================

        quality, _ = model(
            image_tensor,
            feature_tensor,
        )


        quality_score = (
            quality[0]
        )


        # =================================================
        # BACKWARD
        # =================================================

        quality_score.backward()


        if not activations:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )


        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )


        # =================================================
        # TENSOR → NUMPY
        # =================================================

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


        # =================================================
        # GLOBAL AVERAGE POOLING
        # =================================================

        weights = np.mean(
            gradient,
            axis=(1, 2),
        )


        # =================================================
        # WEIGHTED ACTIVATION MAP
        # =================================================

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


        # =================================================
        # RELU
        # =================================================

        cam = np.maximum(
            cam,
            0,
        )


        # =================================================
        # NORMALIZATION
        # =================================================

        maximum = cam.max()


        if maximum > 0:

            cam = (
                cam
                / maximum
            )


        # =================================================
        # RESIZE
        # =================================================

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


        # =================================================
        # 0 → 255
        # =================================================

        heatmap = np.uint8(
            cam * 255
        )


        # =================================================
        # APPLY COLOR MAP
        # =================================================

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )


        # =================================================
        # OVERLAY
        # =================================================

        overlay = cv2.addWeighted(
            original_image,
            0.55,
            heatmap,
            0.45,
            0,
        )


        # =================================================
        # SAVE
        # =================================================

        output_dir = Path(
            output_dir
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        filename = (
            "gradcam_"
            + str(
                np.random.randint(
                    100000,
                    999999,
                )
            )
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