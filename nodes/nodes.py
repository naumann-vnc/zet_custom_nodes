import re
import numpy as np
import torch
from math import gcd
import cv2
from scipy.ndimage import gaussian_filter
from skimage import graph, segmentation, filters, color, filters

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

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge_image"
    CATEGORY = "image/nedzet-nodes"

    def merge_image(self, image: torch.Tensor, compactness: int, n_segments: int, threshold: float):

        for img in image:  # Process each image in the batch
            img_np = img.cpu().numpy()

            # Compute edges using the Sobel filter
            edges = filters.sobel(color.rgb2gray(img_np))

            # Perform SLIC segmentation
            labels = segmentation.slic(
                img_np, compactness=compactness, n_segments=n_segments, start_label=1
            )

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
        return (torch_out,)
