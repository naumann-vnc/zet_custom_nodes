import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
from math import gcd
import cv2


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# Common aspect ratios with their names
common_aspect_ratios = {
    (1, 1): "1:1",
    (4, 3): "4:3",
    (16, 9): "16:9",
    (3, 2): "3:2",
    (21, 9): "21:9",
    (16, 10): "16:10",
    (5, 4): "5:4",
}


# Function to calculate the greatest common divisor (for reducing the ratio)
def calculate_aspect_ratio(width, height):
    divisor = gcd(width, height)
    return width // divisor, height // divisor


# Function to find the closest common aspect ratio
def closest_aspect_ratio(aspect_ratio):
    closest_ratio = None
    closest_diff = float("inf")

    for ratio in common_aspect_ratios:
        # Calculate the absolute difference between ratios
        diff = abs(aspect_ratio[0] / aspect_ratio[1] - ratio[0] / ratio[1])
        if diff < closest_diff:
            closest_diff = diff
            closest_ratio = ratio
    return closest_ratio


# Function to resize an image using OpenCV
def resize_image_to_aspect_ratio(image: np.ndarray, width: int, height: int, closest_ratio):
    closest_width_ratio, closest_height_ratio = closest_ratio

    # Decide whether to adjust width or height to match the closest aspect ratio
    if width / height >= closest_width_ratio / closest_height_ratio:
        # Fix height and adjust width
        new_height = height
        new_width = int((closest_width_ratio / closest_height_ratio) * new_height)
    else:
        # Fix width and adjust height
        new_width = width
        new_height = int((closest_height_ratio / closest_width_ratio) * new_width)

    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return resized_image


# Function to crop the image to fit the closest aspect ratio
def crop_image_to_aspect_ratio(image: np.ndarray, width: int, height: int, closest_ratio):
    closest_width_ratio, closest_height_ratio = closest_ratio

    # Calculate the target aspect ratio width and height
    target_aspect_ratio = closest_width_ratio / closest_height_ratio

    # Determine whether to crop based on width or height
    if width / height >= target_aspect_ratio:
        # Crop width, keeping the height
        new_width = int(target_aspect_ratio * height)
        crop_x = (width - new_width) // 2  # Calculate the x offset for center cropping
        cropped_image = image[:, crop_x:crop_x + new_width]
    else:
        # Crop height, keeping the width
        new_height = int(width / target_aspect_ratio)
        crop_y = (height - new_height) // 2  # Calculate the y offset for center cropping
        cropped_image = image[crop_y:crop_y + new_height, :]

    return cropped_image


# Function to ensure the image resolution does not exceed the max_resolution
def limit_resolution(image: np.ndarray, max_resolution: int):
    height, width = image.shape[:2]

    # If both dimensions are within the limit, return the image as is
    if max(width, height) <= max_resolution:
        return image

    # Calculate the scaling factor to reduce the image size
    scaling_factor = max_resolution / max(width, height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return resized_image


class ResizeImageTargetingAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop": ("BOOLEAN",),
                "max_resolution": ("INT", {"default": 0, "min": 0, "step": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_image_targeting_aspect_ratio"
    CATEGORY = "image/nedzet-nodes"

    def resize_image_targeting_aspect_ratio(self, image: torch.Tensor, crop: bool, max_resolution: int):
        # Get the width and height from the image object
        batch_size, height, width, _ = image.shape

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            # modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # Calculate the aspect ratio of the image
            image_aspect_ratio = calculate_aspect_ratio(width, height)

            # Find the closest common aspect ratio
            closest_ratio = closest_aspect_ratio(image_aspect_ratio)

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
            resized_image = resize_image_to_aspect_ratio(
                tensor_image, width, height, closest_ratio
            )

        # Apply max_resolution if defined
        if max_resolution is not None and max_resolution !=0:
            resized_image = limit_resolution(resized_image, max_resolution)

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
