import argparse
import os
import numpy as np
from PIL import Image
import cv2

# Transformation matrices and utility functions
def rgb_to_lms():
    return np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]]).T

def lms_to_rgb():
    return np.array([[0.0809, -0.1305, 0.1167],
                     [-0.0102, 0.0540, -0.1136],
                     [-0.0004, -0.0041, 0.6935]]).T

def lms_protanopia_sim(degree=1.0):
    return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                     [0, 1, 0],
                     [0, 0, 1]]).T

def lms_deutranopia_sim(degree=1.0):
    return np.array([[1, 0, 0],
                     [0.494207 * degree, 1 - degree, 1.24827 * degree],
                     [0, 0, 1]]).T

def lms_tritanopia_sim(degree=1.0):
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [-0.395913 * degree, 0.801109 * degree, 1 - degree]]).T

def load_lms(path):
    img_rgb = np.array(Image.open(path)) / 255
    img_lms = np.dot(img_rgb[:, :, :3], rgb_to_lms())
    return img_lms

def simulate_color_blindness(input_path, output_path, simulate_type, simulate_degree=1.0):
    assert simulate_type in ['protanopia', 'deutranopia', 'tritanopia'], 'Invalid simulate type'

    img_lms = load_lms(input_path)

    if simulate_type == 'protanopia':
        transform = lms_protanopia_sim(degree=simulate_degree)
    elif simulate_type == 'deutranopia':
        transform = lms_deutranopia_sim(degree=simulate_degree)
    elif simulate_type == 'tritanopia':
        transform = lms_tritanopia_sim(degree=simulate_degree)

    img_sim = np.dot(img_lms, transform)
    img_sim = np.uint8(np.dot(img_sim, lms_to_rgb()) * 255)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_sim)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate Images for Color-Blindness')
    parser.add_argument('-input', type=str, required=True, help='Path to input image.')
    parser.add_argument('-output', type=str, required=True, help='Path to save the output image.')
    parser.add_argument('-type', type=str, choices=['protanopia', 'deutranopia', 'tritanopia'], required=True, help='Type of color blindness to simulate.')
    parser.add_argument('-degree', type=float, default=1.0, help='Intensity of the color blindness simulation. Default is 1.0')
    return parser.parse_args()

def main():
    args = parse_args()
    simulate_color_blindness(args.input, args.output, args.type, args.degree)
    print(f'Color blindness simulation completed! Check the output image at {args.output}')

if __name__ == '__main__':
    main()