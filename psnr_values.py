import cv2
import numpy as np
import os

def calculate_psnr(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('100')  # PSNR is infinite if images are identical
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def resize_image(image, target_shape):
    return cv2.resize(image, target_shape)

def calculate_psnr_for_dataset(dataset_path, reference_image_path):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None:
        print(f"Error: Unable to read the reference image from {reference_image_path}")
        return

    print(f"Reference Image Shape: {reference_image.shape}")

    psnr_values = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Error: Unable to read the image from {image_path}")
                continue
            
            # Resize image to match reference image dimensions
            image = resize_image(image, (reference_image.shape[1], reference_image.shape[0]))

            psnr = calculate_psnr(reference_image, image)
            psnr_values.append(psnr)

    return psnr_values

if __name__ == "__main__":
    dataset_path = "D:/ProjectDataset/OwnDataset/Peepal/Train/Healthy"
    reference_image_path = "D:/ProjectDataset/OwnDataset/Peepal/Train/Healthy/IMG_6886(1).JPG"
    
    psnr_results = calculate_psnr_for_dataset(dataset_path, reference_image_path)

    if psnr_results:
        #print("\nList of PSNR Values:")
       # print(psnr_results)
        print((sum(psnr_results)/len(psnr_results)))