import cv2
import gc
import numpy as np
import torch

from pathlib import Path
from torchvision.transforms.functional import to_pil_image

from ml.config import (
    ARTIFACTS_DIR,
    PROCESSED_DIR,
    IMAGE_DIR,
)

from ml.model import ImageQualityNet
from dataset import KonIQDataset


class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model

        self.target_layer = target_layer

    def generate(
        self,
        image,
        features,
    ):

        activations = []
        gradients = []

        def save_activation(module, hook_input, output):
            activations.append(output.detach())

        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_hook = backward_hook = None
        quality = defect_output = weights = cam_tensor = None
        try:
            forward_hook = self.target_layer.register_forward_hook(save_activation)
            backward_hook = self.target_layer.register_full_backward_hook(save_gradient)
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                image.requires_grad_(True)
                quality, defect_output = self.model(image, features)
                quality[0].backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam_tensor = torch.relu((weights * activations[0]).sum(dim=1)).squeeze(0)
            cam = cam_tensor.cpu().numpy()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        finally:
            if forward_hook is not None:
                forward_hook.remove()
            if backward_hook is not None:
                backward_hook.remove()
            activations.clear()
            gradients.clear()
            self.model.zero_grad(set_to_none=True)
            del quality, defect_output, weights, cam_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # LOAD FEATURE STATISTICS
    feature_mean = np.load(
        ARTIFACTS_DIR
        / "feature_mean.npy"
    )

    feature_std = np.load(
        ARTIFACTS_DIR
        / "feature_std.npy"
    )

 
    # LOAD DATASET
    dataset = KonIQDataset(
        PROCESSED_DIR / "test.csv",
        feature_mean,
        feature_std,
        training=False,
    )


    # LOAD MODEL
    model = ImageQualityNet(
        num_features=8,
        num_defects=5,
    ).to(device)

    checkpoint = torch.load(
        ARTIFACTS_DIR
        / "image_quality_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()


    # TARGET LAYER

    target_layer = (
        model.backbone.layer4[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer,
    )

    # FIRST TEST IMAGE

    sample = dataset[0]

    image_tensor = sample[0].unsqueeze(
        0
    ).to(device)

    features = sample[1].unsqueeze(
        0
    ).to(device)

    cam = gradcam.generate(
        image_tensor,
        features,
    )


    # GET ORIGINAL IMAGE
    test_df = __import__(
        "pandas"
    ).read_csv(
        PROCESSED_DIR / "test.csv"
    )

    filename = str(
        test_df.iloc[0]["filename"]
    )

    image_path = (
        IMAGE_DIR
        / filename
    )

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise FileNotFoundError(
            image_path
        )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB,
    )


    # RESIZE CAM
    cam = cv2.resize(
        cam,
        (
            original.shape[1],
            original.shape[0],
        ),
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        0.55 * original
        + 0.45 * heatmap
    )

    overlay = np.uint8(
        np.clip(
            overlay,
            0,
            255,
        )
    )

    output_dir = (
        ARTIFACTS_DIR
        / "gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "gradcam_example.jpg"
    )

    cv2.imwrite(
        str(output_file),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print(
        f"\n✓ Grad-CAM saved:"
        f"\n{output_file}"
    )


if __name__ == "__main__":
    main()
