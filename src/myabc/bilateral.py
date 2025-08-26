'''
Copyright 2025 Sergio Renato Rengifo Lozano

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import cv2
import numpy as np
import math
from scipy.special import gamma

def ABC_bilateral(image, d, sC, sS):
    """Gradiente adaptativo con filtro bilateral."""

    # ----------------------------
    # Funciones auxiliares
    # ----------------------------
    def gcd(a, b):
        return math.gcd(a, b)

    def calculate_area_size(image):
        height, width = image.shape
        gcd_hw = gcd(height, width)
        return height // gcd_hw, width // gcd_hw

    area_height, area_width = calculate_area_size(image)

    def apply_bilateral_filter(image):
        return cv2.bilateralFilter(image.astype(np.float32), d=d, sigmaColor=sC, sigmaSpace=sS)

    def calculate_alpha_grid(image, area_height, area_width):
        normal_gray = image
        bilateral_gray = apply_bilateral_filter(normal_gray)
        prod_image = np.abs(normal_gray * bilateral_gray)

        a_min, a_max = 0.01, 0.99
        alpha_grid = np.zeros_like(prod_image)

        for i in range(0, prod_image.shape[0] - area_height + 1, area_height):
            for j in range(0, prod_image.shape[1] - area_width + 1, area_width):
                area = prod_image[i:i+area_height, j:j+area_width]
                prom_area = np.mean(area)
                alpha = a_min + (a_max - a_min) * prom_area
                alpha_grid[i:i+area_height, j:j+area_width] = alpha

        return alpha_grid

    def kernel_value(alpha):
        Lambda = alpha / (1 - alpha)
        M = 1 - alpha + (alpha / gamma(alpha))
        K = M / (1 - alpha)
        g = gamma(alpha + 2)

        p0 = K * (1 - (Lambda**(2 - alpha)) / g + (Lambda**2) * (2**(-2 * alpha)) / g)
        p1 = K * (2 * Lambda * (2**(-2 * alpha) - 1) / g + (2 * Lambda**2) * (1 - 2**(-2 * alpha)) / g)
        p2 = K * (-1 + Lambda * (2 - 2**(-alpha)) / g - (Lambda**2) * (2**(-2 * alpha) - 2) / g)

        return p0, p1, p2

    def kernel(alpha):
        p0, p1, p2 = kernel_value(alpha)
        h_x = np.array([[-p0, 0, p0], [-p1, 0, p1], [-p2, 0, p2]])
        h_y = np.array([[-p0, -p1, -p2], [0, 0, 0], [p0, p1, p2]])
        return h_x, h_y

    def compute_gradient(image, alpha_grid):
        grad_x = np.zeros_like(image, dtype=np.float32)
        grad_y = np.zeros_like(image, dtype=np.float32)

        for x in range(1, image.shape[0] - 1):
            for y in range(1, image.shape[1] - 1):
                alpha = alpha_grid[x, y]
                h_x, h_y = kernel(alpha)
                patch = image[x-1:x+2, y-1:y+2]
                grad_x[x, y] = np.sum(patch * h_x)
                grad_y[x, y] = np.sum(patch * h_y)

        grad_x_filtered = cv2.bilateralFilter(grad_x, d=d, sigmaColor=sC, sigmaSpace=sS, borderType=cv2.BORDER_REPLICATE)
        grad_y_filtered = cv2.bilateralFilter(grad_y, d=d, sigmaColor=sC, sigmaSpace=sS, borderType=cv2.BORDER_REPLICATE)

        grad_magnitude = np.sqrt(grad_x_filtered**2 + grad_y_filtered**2)

        grad_x_vis = cv2.normalize(grad_x_filtered, None, 0, 255, cv2.NORM_MINMAX)
        grad_y_vis = cv2.normalize(grad_y_filtered, None, 0, 255, cv2.NORM_MINMAX)
        grad_mag_vis = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return grad_x_vis, grad_y_vis, grad_mag_vis

    alpha_grid = calculate_alpha_grid(image, area_height, area_width)
    return compute_gradient(image, alpha_grid) + (alpha_grid,)
