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

def ABC_var(image, p, q, d, sC, sS):
    """
    Gradiente adaptativo basado en la varianza local.
    
    Parámetros
    ----------
    image : np.ndarray
        Imagen de entrada en escala de grises, normalizada (0-1 o 0-255).
    p : int, opcional
        Altura de la ventana local. Default = 3.
    q : int, opcional
        Anchura de la ventana local. Default = 3.
    d : int, opcional
        Diámetro del filtro bilateral. Default = 10.
    sC : float, opcional
        Sigma Color para filtro bilateral. Default = 80.
    sS : float, opcional
        Sigma Space para filtro bilateral. Default = 100.

    Retorna
    -------
    grad_x_vis : np.ndarray
        Gradiente en X normalizado (0-255).
    grad_y_vis : np.ndarray
        Gradiente en Y normalizado (0-255).
    grad_mag_vis : np.ndarray
        Magnitud del gradiente normalizada (0-255).
    alpha_map : np.ndarray
        Mapa de valores de α calculados localmente.
    """

    # ----------------------------
    # Función auxiliar: alpha local
    # ----------------------------
    def compute_alpha_local(image_norm, x, y, p, q):
        half_p, half_q = p // 2, q // 2
        x1, x2 = max(0, x-half_p), min(image_norm.shape[0], x+half_p+1)
        y1, y2 = max(0, y-half_q), min(image_norm.shape[1], y+half_q+1)
        
        window = image_norm[x1:x2, y1:y2]
        
        mu = np.mean(window)
        
        sigma = np.mean(window**2) - mu**2
        
        local_var = window**2 - mu**2
        sigma_min = np.min(local_var)
        sigma_max = np.max(local_var)
        
        if sigma_max - sigma_min < 1e-8:
            return 0.0
        
        alpha = np.sqrt((sigma - sigma_min) / (sigma_max - sigma_min))
        return alpha
        
    def kernel_value(alpha):
        Lambda = alpha / (1 - alpha)
        M = 1 - alpha + (alpha / gamma(alpha))
        K = M / (1 - alpha)
        
        g = gamma(alpha + 2)
        
        p0 = K * (1 - (Lambda ** (2 - alpha)) / g + (Lambda ** 2) * (2 ** (-2 * alpha)) / g)
        p1 = K * (2 * Lambda * (2 ** (-2 * alpha) - 1) / g + (2 * Lambda ** 2) * (1 - 2 ** (-2 * alpha)) / g)
        p2 = K * (Lambda * (2 - 2 ** (-alpha)) / g - (Lambda ** 2) * (2 ** (-2 * alpha) - 2) / g)
        
        return p0, p1, p2
    
    def kernel(alpha):
        p0, p1, p2 = kernel_value(alpha)
        h_x = np.array([[-p0, 0, p0], [-p1, 0, p1], [-p2, 0, p2]])
        h_y = np.array([[-p0, -p1, -p2], [0, 0, 0], [p0, p1, p2]])
        return h_x, h_y

    # ----------------------------
    # Cálculo del gradiente
    # ----------------------------
    def compute_gradient(image_norm, p, q):
        grad_x = np.zeros_like(image_norm, dtype=np.float32)
        grad_y = np.zeros_like(image_norm, dtype=np.float32)
        alpha_map = np.zeros_like(image_norm, dtype=np.float32)  # aquí guardaremos los alphas
        
        for x in range(1, image_norm.shape[0] - 1):
            for y in range(1, image_norm.shape[1] - 1):
                alpha = compute_alpha_local(image_norm, x, y, p, q)
                alpha_map[x, y] = alpha   # guardamos el valor en la matriz
                
                h_x, h_y = kernel(alpha)
                
                patch = image_norm[x-1:x+2, y-1:y+2]
                
                grad_x[x, y] = np.sum(patch * h_x)
                grad_y[x, y] = np.sum(patch * h_y)
        
        grad_x_filtered = cv2.bilateralFilter(grad_x, d=d, sigmaColor=sC, sigmaSpace=sS, 		borderType=cv2.BORDER_REPLICATE)
        grad_y_filtered = cv2.bilateralFilter(grad_y, d=d, sigmaColor=sC, sigmaSpace=sS, 		borderType=cv2.BORDER_REPLICATE)
        
        grad_magnitude = np.sqrt(grad_x_filtered**2 + grad_y_filtered**2)
        
        grad_x_vis = cv2.normalize(grad_x_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_y_vis = cv2.normalize(grad_y_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_mag_vis = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return grad_x_vis, grad_y_vis, grad_mag_vis, alpha_map

    grad_x_vis, grad_y_vis, grad_mag_vis, alpha_map = compute_gradient(image, p, q)
    return grad_x_vis, grad_y_vis, grad_mag_vis, alpha_map
	
