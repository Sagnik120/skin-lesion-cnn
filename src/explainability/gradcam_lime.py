"""
Explainability — Grad-CAM and LIME for skin lesion predictions.

Grad-CAM:
  Visualizes which pixels influenced the model's decision by computing
  gradients of the class score with respect to the last convolutional layer.

LIME (Local Interpretable Model-agnostic Explanations):
  Segments the image into superpixels, perturbs them, and fits a local
  linear model to explain a single prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation using PyTorch hooks.

    Usage:
        cam = GradCAM(model, target_layer=model.get_last_conv_layer())
        heatmap = cam(image_tensor, class_idx)
        cam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fwd = self.target_layer.register_forward_hook(forward_hook)
        self._bwd = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            image_tensor: (1, C, H, W) normalized tensor on model's device
            class_idx: target class; if None, uses argmax (predicted class)

        Returns:
            heatmap: (H, W) float array in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Resize to input image size
        h, w = image_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_image: (H, W, 3) uint8 RGB array
        heatmap: (H, W) float in [0, 1]
        alpha: transparency of the heatmap overlay

    Returns:
        overlaid: (H, W, 3) uint8 RGB array
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid      = (original_image * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return overlaid


# ── LIME ──────────────────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    Wrapper around lime.lime_image for skin lesion explanations.

    Usage:
        explainer = LIMEExplainer(model, device)
        explanation = explainer.explain(pil_image, class_idx)
        explainer.plot(pil_image, explanation, class_idx, save_path)
    """

    def __init__(self, model: torch.nn.Module, device: str, image_size: int = 224):
        self.model      = model
        self.device     = device
        self.image_size = image_size

        try:
            from lime import lime_image
            self._lime_image = lime_image
        except ImportError:
            raise ImportError("LIME not installed. Run: pip install lime")

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction function for LIME — takes (N, H, W, 3) uint8 → (N, num_classes)."""
        from torchvision import transforms
        from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.model.eval()
        batch = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8))) for img in images
        ]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(
        self,
        pil_image: Image.Image,
        class_idx: int | None = None,
        num_samples: int = 1000,
    ):
        """Run LIME explanation for one image."""
        img_array = np.array(pil_image.resize((self.image_size, self.image_size)))
        explainer = self._lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            self._predict_fn,
            top_labels=7,
            hide_color=0,
            num_samples=num_samples,
        )
        if class_idx is None:
            class_idx = explanation.top_labels[0]
        return explanation, class_idx

    def plot(
        self,
        pil_image: Image.Image,
        explanation,
        class_idx: int,
        save_path: str | Path,
        model_name: str = "",
    ):
        """Save a LIME explanation plot."""
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        from skimage.segmentation import mark_boundaries

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.array(pil_image.resize((self.image_size, self.image_size))))
        axes[0].set_title("Original image"); axes[0].axis("off")
        axes[1].imshow(mark_boundaries(temp / 255.0, mask))
        axes[1].set_title(f"LIME — top superpixels ({model_name})"); axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"LIME plot saved → {save_path}")


# ── Batch Grad-CAM visualization ───────────────────────────────────────────────

def visualize_gradcam_batch(
    model,
    loader,
    device: str,
    save_dir: str | Path,
    n_images: int = 8,
    model_name: str = "",
):
    """
    Run Grad-CAM on the first `n_images` from a DataLoader and save a grid.
    """
    from src.data.dataset import IDX_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)

    cam = GradCAM(model, model.get_last_conv_layer())
    model.eval()

    collected = 0
    fig, axes = plt.subplots(n_images, 3, figsize=(10, n_images * 3))
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=13)

    for images, labels, _ in loader:
        for i in range(len(images)):
            if collected >= n_images:
                break

            img_tensor = images[i:i+1].to(device)
            label      = labels[i].item()

            # De-normalise for display
            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean).clip(0, 1)
            img_uint8 = (img_np * 255).astype(np.uint8)

            heatmap, pred_idx = cam(img_tensor)
            overlaid = overlay_gradcam(img_uint8, heatmap)

            axes[collected, 0].imshow(img_uint8)
            axes[collected, 0].set_title(f"True: {IDX_TO_CLASS[label]}", fontsize=8)
            axes[collected, 0].axis("off")

            axes[collected, 1].imshow(heatmap, cmap="jet")
            axes[collected, 1].set_title("Heatmap", fontsize=8)
            axes[collected, 1].axis("off")

            axes[collected, 2].imshow(overlaid)
            axes[collected, 2].set_title(f"Overlay (pred: {IDX_TO_CLASS[pred_idx]})", fontsize=8)
            axes[collected, 2].axis("off")

            collected += 1

        if collected >= n_images:
            break

    cam.remove_hooks()
    plt.tight_layout()
    save_path = save_dir / f"gradcam_{model_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM grid saved → {save_path}")
    return save_path
