import cv2
import os
import numpy as np
import argparse

def apply_gaussian_blur(image_path, output_path, blur_amount):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, blur_amount, cv2.BORDER_DEFAULT)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)

def apply_directional_blur(image_path, output_path, kernel_size, angle):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Create the directional blur kernel
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = np.diag(np.ones(kernel_size))
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    
    # Apply directional blur
    blurred_image = cv2.filter2D(image, -1, kernel)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply Gaussian or directional blur to an image.")
    parser.add_argument('blur_type', choices=['gaussian', 'directional'], help="Type of blur to apply.")
    parser.add_argument('input_image', help="Path to the input image.")
    parser.add_argument('output_image', help="Path to save the blurred image.")
    parser.add_argument('--blur_amount', type=int, nargs=2, default=[21, 21], help="Kernel size for Gaussian blur.")
    parser.add_argument('--kernel_size', type=int, default=21, help="Kernel size for directional blur.")
    parser.add_argument('--angle', type=float, default=45, help="Angle for directional blur.")
    
    args = parser.parse_args()
    
    if args.blur_type == 'gaussian':
        apply_gaussian_blur(args.input_image, args.output_image, tuple(args.blur_amount))
    elif args.blur_type == 'directional':
        apply_directional_blur(args.input_image, args.output_image, args.kernel_size, args.angle)

if __name__ == "__main__":
    main()