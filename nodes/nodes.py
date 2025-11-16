import re
import numpy as np
import torch
from math import gcd
import cv2
from scipy.ndimage import gaussian_filter
from skimage import graph, segmentation, filters, color
from torchvision import transforms
from PIL import Image

# Target aspect ratios and corresponding resolutions
target_resolutions = {
    (1, 1): (1024, 1024),
    (3, 4): (896, 1152),
    (5, 8): (832, 1216),
    (9, 16): (768, 1344),
    (9, 21): (640, 1536),
    (4, 3): (1152, 832),
    (16, 9): (1344, 768),
    (21, 9): (1536, 640),
}


# Function to calculate the greatest common divisor (for reducing the ratio)
def calculate_aspect_ratio(width, height):
    divisor = gcd(width, height)
    return width // divisor, height // divisor


# Function to find the closest target aspect ratio
def closest_aspect_ratio(width, height):
    aspect_ratio = width / height
    closest_ratio = None
    closest_diff = float("inf")

    for ratio in target_resolutions.keys():
        target_aspect_ratio = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect_ratio)
        if diff < closest_diff:
            closest_diff = diff
            closest_ratio = ratio

    return closest_ratio


# Function to resize an image to the target resolution
def resize_image_to_target_resolution(image: np.ndarray, target_resolution):
    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(
        image, target_resolution, interpolation=cv2.INTER_LANCZOS4
    )
    return resized_image


# Function to crop the image to fit the closest aspect ratio
def crop_image_to_aspect_ratio(
    image: np.ndarray, width: int, height: int, closest_ratio
):
    closest_width_ratio, closest_height_ratio = closest_ratio

    # Calculate the target aspect ratio width and height
    target_aspect_ratio = closest_width_ratio / closest_height_ratio

    # Determine whether to crop based on width or height
    if width / height >= target_aspect_ratio:
        # Crop width, keeping the height
        new_width = int(target_aspect_ratio * height)
        crop_x = (width - new_width) // 2  # Calculate the x offset for center cropping
        cropped_image = image[:, crop_x : crop_x + new_width]
    else:
        # Crop height, keeping the width
        new_height = int(width / target_aspect_ratio)
        crop_y = (
            height - new_height
        ) // 2  # Calculate the y offset for center cropping
        cropped_image = image[crop_y : crop_y + new_height, :]

    return cropped_image


class ResizeImageTargetingAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "crop": ("BOOLEAN",)}}

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_image_targeting_aspect_ratio"
    CATEGORY = "image/nedzet-nodes"

    def resize_image_targeting_aspect_ratio(self, image: torch.Tensor, crop: bool):
        # Get the width and height from the image object
        batch_size, height, width, _ = image.shape

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            # modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # Find the closest aspect ratio
            closest_ratio = closest_aspect_ratio(width, height)
            # Get the target resolution for the closest aspect ratio
            target_resolution = target_resolutions.get(closest_ratio, (width, height))

            # Get the name of the closest common aspect ratio
            # closest_ratio_name = common_aspect_ratios[closest_ratio]

            # Resize the image to the closest aspect ratio
            if crop:
                # Crop the image to fit the closest aspect ratio
                cropped_image = crop_image_to_aspect_ratio(
                    tensor_image, width, height, closest_ratio
                )
                resized_image = resize_image_to_target_resolution(
                    cropped_image, target_resolution
                )
            else:
                # Resize the image to fit the closest aspect ratio
                resized_image = resize_image_to_target_resolution(
                    tensor_image, target_resolution
                )

        resized_image = torch.from_numpy(resized_image).unsqueeze(0)

        return (resized_image,)


class LaplacianFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "smoothness": ("INT", {"default": 2, "min": 0, "max": 10}),
                "min_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1, "step": 0.01},
                ),
                "max_threshold": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "max": 1, "step": 0.01},
                ),
                "absolute_value": ("BOOLEAN",),
                "negative_colors": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "apply_laplacian_filter"
    CATEGORY = "image/nedzet-nodes"

    def apply_laplacian_filter(
        self,
        image: torch.Tensor,
        smoothness: int,
        min_threshold: float,
        max_threshold: float,
        absolute_value: bool,
        negative_colors: bool,
    ):

        # Convert tensor to a numpy array with compatible type
        image_np = (
            image.squeeze().numpy().astype(np.float32) * 255
        )  # Convert tensor to numpy and scale for OpenCV

        # Apply Gaussian smoothing (smoothness)
        if smoothness > 0:
            image_np = gaussian_filter(image_np, sigma=smoothness)

        # Apply Laplacian filter with proper format
        laplacian_np = cv2.Laplacian(image_np, cv2.CV_32F, ksize=3)

        # Apply absolute value if specified
        if absolute_value:
            laplacian_np = np.abs(laplacian_np)

        # Normalize Laplacian output to range 0-1
        laplacian_normalized = cv2.normalize(laplacian_np, None, 0, 1, cv2.NORM_MINMAX)

        # Apply min and max thresholds
        laplacian_normalized = np.clip(
            laplacian_normalized, min_threshold, max_threshold
        )

        # Rescale to 0-1 based on thresholds
        laplacian_normalized = (laplacian_normalized - min_threshold) / (
            max_threshold - min_threshold
        )
        laplacian_normalized = np.clip(laplacian_normalized, 0, 1)

        # Invert colors if specified
        if negative_colors:
            laplacian_normalized = 1 - laplacian_normalized

        # Convert back to tensor
        laplacian_tensor = torch.tensor(
            laplacian_normalized, dtype=torch.float32
        ).unsqueeze(
            0
        )  # Convert back to tensor

        return (laplacian_tensor,)


class BlendMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_batch": ("MASK",),
                "target_mask": ("MASK",),
                "operation": (["add", "subtract"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_and_subtract_masks"
    CATEGORY = "image/nedzet-nodes"

    def combine_and_subtract_masks(self, mask_batch, target_mask, operation):
        # Sum all the masks in the batch
        if mask_batch.ndim > 2:
            summed_mask = torch.sum(mask_batch, dim=0)
        else:
            summed_mask = torch.clamp(
                torch.sum(mask_batch.unsqueeze(1), dim=0), 0, 255
            ).squeeze(1)

        # Apply the specified operation with the regular mask and summed mask
        if operation == "add":
            if target_mask.ndim > 2:
                result_mask = target_mask + summed_mask
            else:
                result_mask = torch.clamp(
                    target_mask.unsqueeze(1) + summed_mask, 0, 255
                ).squeeze(1)
        elif operation == "subtract":
            if target_mask.ndim > 2:
                result_mask = target_mask - summed_mask
            else:
                result_mask = torch.clamp(
                    target_mask.unsqueeze(1) - summed_mask, 0, 255
                ).squeeze(1)
        else:
            raise ValueError("Invalid operation. Use 'add' or 'subtract'.")

        return (result_mask,)


class TagBlacklist:

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": False}),
                "words_to_remove": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remove_words"
    CATEGORY = "utils/nedzet-nodes"

    def remove_words(self, text, words_to_remove):
        # Escape any special regex characters in words to remove
        words_pattern = "|".join(
            re.escape(word.strip()) for word in words_to_remove.split(",")
        )

        # Create regex to match each word or phrase, with optional leading commas or spaces
        cleaned_text = re.sub(r",?\s*(" + words_pattern + r")", "", text).strip()

        # Remove any leading/trailing commas or extra spaces that might remain
        cleaned_text = re.sub(r"\s*,\s*", ", ", cleaned_text).strip(", ")

        return (cleaned_text,)


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.
    """
    default = {"weight": 0.0, "count": 0}

    count_src = graph[src].get(n, default)["count"]
    count_dst = graph[dst].get(n, default)["count"]

    weight_src = graph[src].get(n, default)["weight"]
    weight_dst = graph[dst].get(n, default)["weight"]

    count = count_src + count_dst
    return {
        "count": count,
        "weight": (count_src * weight_src + count_dst * weight_dst) / count,
    }


def merge_boundary(graph, src, dst):
    """Callback called before merging 2 nodes.

    In this case, we don't need to do any computation here.
    """
    pass


class HierarchicalMergingofRegionBoundaryRAGs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "compactness": ("INT", {"default": 30, "min": 1, "max": 100}),
                "n_segments": ("INT", {"default": 400, "min": 1}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
    )
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge_image"
    CATEGORY = "image/nedzet-nodes"

    def merge_image(
        self, image: torch.Tensor, compactness: int, n_segments: int, threshold: float
    ):
        num_segments_per_image = []

        for img in image:  # Process each image in the batch
            img_np = img.cpu().numpy()

            # Compute edges using the Sobel filter
            edges = filters.sobel(color.rgb2gray(img_np))

            # Perform SLIC segmentation
            labels = segmentation.slic(
                img_np, compactness=compactness, n_segments=n_segments, start_label=1
            )

            # Get the unique segment labels
            unique_labels = np.unique(labels)
            num_segments = len(unique_labels)

            # Store the number of segments
            num_segments_per_image.append(num_segments)

            # Create the Region Adjacency Graph (RAG) using boundary weights
            g = graph.rag_boundary(labels, edges)

            # Perform hierarchical merging
            labels2 = graph.merge_hierarchical(
                labels,
                g,
                thresh=threshold,
                rag_copy=False,
                in_place_merge=True,
                merge_func=merge_boundary,
                weight_func=weight_boundary,
            )

            # Convert labels back to an image
            out = color.label2rgb(labels2, img_np, kind="avg", bg_label=0)

        torch_out = torch.from_numpy(out).unsqueeze(0)
        return (
            torch_out,
            num_segments_per_image,
        )


class MergeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination_image": ("IMAGE",),
                "pasted_image": ("IMAGE",),
                "x": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
                "y": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
                "anchor": (
                    ["top-right", "top-left", "bottom-left", "bottom-right", "center"],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "image/nedzet-nodes"

    def merge_images(
        self,
        destination_image: torch.Tensor,
        pasted_image: torch.Tensor,
        x: int,
        y: int,
        anchor: str,
    ):

        batch_size, height, width, channels = pasted_image.shape

        # Placeholder for results
        result_batch = []

        for b in range(batch_size):
            # Convert tensors to numpy arrays for PIL compatibility
            dest_np = (destination_image[b].cpu().numpy() * 255).astype("uint8")
            pasted_np = (pasted_image[b].cpu().numpy() * 255).astype("uint8")

            # Ensure the images are in RGBA mode (to support transparency)
            pil_destination = Image.fromarray(dest_np, mode="RGBA" if channels == 4 else "RGB")
            pil_pasted = Image.fromarray(pasted_np, mode="RGBA" if channels == 4 else "RGB")

            # Get dimensions of the destination and pasted image
            dest_width, dest_height = pil_destination.size
            pasted_width, pasted_height = pil_pasted.size

            # Adjust coordinates based on the anchor
            if anchor == "top-right":
                x -= pasted_width
            elif anchor == "bottom-left":
                y -= pasted_height
            elif anchor == "bottom-right":
                x -= pasted_width
                y -= pasted_height
            elif anchor == "center":
                x = (dest_width - pasted_width) // 2
                y = (dest_height - pasted_height) // 2

            # Copy the destination image to avoid modifying the original
            result_image = pil_destination.copy()

            # Paste the image with the alpha channel as a mask
            if pil_pasted.mode == "RGBA":
                result_image.paste(pil_pasted, (x, y), pil_pasted.split()[-1])  # Use alpha as mask
            else:
                result_image.paste(pil_pasted, (x, y))

            # Convert back to a tensor, ensuring that the alpha channel is preserved
            result_np = np.array(result_image)
            result_tensor = torch.from_numpy(result_np).float() / 255.0  # Normalize to [0, 1]

            # Ensure the tensor shape is [H, W, C]
            result_batch.append(result_tensor)

        # Stack all batch images into a single tensor with shape [B, H, W, C]
        result_tensor = torch.stack(result_batch, dim=0)

        return (result_tensor,)


# Detection method from https://github.com/CHEREF-Mehdi/SkinDetection
class SkinToneMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tone_mode": (["average", "median"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")  # (mask, skin_tone_image)
    RETURN_NAMES = ("skin_mask", "skin_tone")
    FUNCTION = "process"
    CATEGORY = "Image/Mask"

    def process(self, images: torch.Tensor, tone_mode: str):
        device = images.device
        images_np = images.detach().cpu().numpy()

        # Normalize to [0,255] if needed
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)

        batch_masks = []
        batch_tones = []

        for img in images_np:
            # Convert from RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # --- HSV MASK ---
            img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
            HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            # --- YCrCb MASK ---
            img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
            YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            # --- Combined mask ---
            global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
            global_mask = cv2.medianBlur(global_mask, 3)
            global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

            # 3-channel mask for Comfy
            mask_3ch = cv2.cvtColor(global_mask, cv2.COLOR_GRAY2RGB)
            mask_tensor = torch.from_numpy(mask_3ch.astype(np.float32) / 255.0)

            # ------------------------------------------------------
            # SKIN TONE EXTRACTION
            # ------------------------------------------------------
            skin_pixels = img[global_mask > 0]  # extract (N,3) rgb pixels

            if len(skin_pixels) == 0:
                # No skin detected → return black
                tone = np.array([0, 0, 0], dtype=np.uint8)
            else:
                if tone_mode == "median":
                    tone = np.median(skin_pixels, axis=0).astype(np.uint8)
                else:
                    tone = np.mean(skin_pixels, axis=0).astype(np.uint8)

            # Create a solid color image
            H, W, C = img.shape
            skin_tone_img = np.zeros((H, W, 3), dtype=np.uint8)
            skin_tone_img[:] = tone  # fill with tone

            tone_tensor = torch.from_numpy(skin_tone_img.astype(np.float32) / 255.0)

            batch_masks.append(mask_tensor)
            batch_tones.append(tone_tensor)

        batch_masks = torch.stack(batch_masks).to(device)
        batch_tones = torch.stack(batch_tones).to(device)

        return (batch_masks, batch_tones)