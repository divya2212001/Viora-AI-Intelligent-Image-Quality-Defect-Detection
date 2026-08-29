import cv2
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

        self.activations = None

        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output,
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0]
        )

    def generate(
        self,
        image,
        features,
    ):

        self.model.zero_grad()

        quality, _ = self.model(
            image,
            features,
        )

        quality.backward()

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * activations
        ).sum(
            dim=1
        )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze(
            0
        ).detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam


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