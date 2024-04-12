import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim

def calculate_ssim(original_image, compressed_image):
    ssim_value = compare_ssim(original_image, compressed_image)
    return ssim_value

def resize_image(image, target_shape):
    return cv2.resize(image, target_shape)

def calculate_ssim_for_dataset(dataset_path, reference_image_path):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None:
        print(f"Error: Unable to read the reference image from {reference_image_path}")
        return

    print(f"Reference Image Shape: {reference_image.shape}")

    ssim_values = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Error: Unable to read the image from {image_path}")
                continue
            
            # Resize image to match reference image dimensions
            image = resize_image(image, (reference_image.shape[1], reference_image.shape[0]))

            ssim = calculate_ssim(reference_image, image)
            ssim_values.append(ssim)

    return ssim_values

if __name__ == "__main__":
    dataset_path = "D:/ProjectDataset/OwnDataset/Peepal/Train/Healthy"
    reference_image_path = "D:/ProjectDataset/OwnDataset/Peepal/Train/Healthy/photo1709822097 (6).jpeg"
    ssim_results = calculate_ssim_for_dataset(dataset_path, reference_image_path)

    if ssim_results:
        #print("\nList of SSIM Values:")
        #print(ssim_results)
        print((sum(ssim_results)/len(ssim_results)))







