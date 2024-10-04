import numpy as np
import torch
from math import gcd
import cv2


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


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
                resized_image = crop_image_to_aspect_ratio(
                    tensor_image, width, height, closest_ratio
                )
                # print(f"Tutorial Text")
            else:
                # Resize the image to fit the closest aspect ratio
                resized_image = resize_image_to_target_resolution(
                    tensor_image, target_resolution
                )

        resized_image = torch.from_numpy(resized_image).unsqueeze(0)

        return (resized_image,)


class PrintHelloWorld:

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "Hello World"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_text"
    OUTPUT_NODE = True
    CATEGORY = "🧩 Tutorial Nodes"

    def print_text(self, text):

        print(f"Tutorial Text : {text}")

        return {}
