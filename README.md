# myabc

`myabc` is a Python package for adaptive edge detection in grayscale images. It implements two methods based on the **Atangana--Baleanu fractional derivative in the Caputo sense**: one driven by bilateral filtering and another driven by local variance.

## Features

- Two adaptive edge detection methods:
  - `ABC_bilateral`: builds a spatial `alpha` map using a bilateral filter.
  - `ABC_var`: computes `alpha` locally from a variance-based window.
- Interpretable outputs:
  - horizontal gradient
  - vertical gradient
  - gradient magnitude
  - adaptive parameter map (`alpha_grid` or `alpha_map`)
- Modern Python packaging with `pyproject.toml`
- Compatible with Python 3.8+

## Installation

### From the repository

```bash
git clone https://github.com/rnloz26/AdaptativeEdgeDetectorsABC.git
cd AdaptativeEdgeDetectorsABC
pip install .
```

### From the local wheel

```bash
pip install dist/myabc-0.1.0-py3-none-any.whl
```

## Requirements

The package depends on:

- `numpy`
- `opencv-python`
- `scipy`
- `matplotlib`

These dependencies are automatically installed with `pip install .`.

## Project structure

```text
AdaptativeEdgeDetectorsABC/
├── dist/
│   ├── myabc-0.1.0-py3-none-any.whl
│   └── myabc-0.1.0.tar.gz
├── src/
│   └── myabc/
│       ├── __init__.py
│       ├── bilateral.py
│       └── var.py
├── LICENSE.txt
└── pyproject.toml
```

## Available methods

### 1. `ABC_bilateral(image, d, sC, sS)`

Computes an adaptive gradient using a bilateral filter to construct a spatial `alpha` map.

**Parameters**

- `image`: 2D grayscale image as a `numpy.ndarray`
- `d`: diameter of the bilateral filter
- `sC`: `sigmaColor` for the bilateral filter
- `sS`: `sigmaSpace` for the bilateral filter

**Returns**

- `grad_x_vis`: normalized horizontal gradient
- `grad_y_vis`: normalized vertical gradient
- `grad_mag_vis`: normalized gradient magnitude
- `alpha_grid`: spatial map of the adaptive parameter `alpha`

### 2. `ABC_var(image, p, q, d, sC, sS)`

Computes an adaptive gradient using a local `alpha` estimation based on variance over a `p × q` window.

**Parameters**

- `image`: 2D grayscale image as a `numpy.ndarray`
- `p`: local window height
- `q`: local window width
- `d`: diameter of the bilateral filter
- `sC`: `sigmaColor` for the bilateral filter
- `sS`: `sigmaSpace` for the bilateral filter

**Returns**

- `grad_x_vis`: normalized horizontal gradient
- `grad_y_vis`: normalized vertical gradient
- `grad_mag_vis`: normalized gradient magnitude
- `alpha_map`: local `alpha` map

## Example usage

```python
import cv2
import matplotlib.pyplot as plt
from myabc.bilateral import ABC_bilateral
from myabc.var import ABC_var

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

# Bilateral-based method
grad_x_b, grad_y_b, grad_mag_b, alpha_b = ABC_bilateral(img, d=9, sC=80, sS=80)

# Variance-based method
grad_x_v, grad_y_v, grad_mag_v, alpha_v = ABC_var(img, p=3, q=3, d=9, sC=80, sS=80)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(grad_mag_b, cmap="gray")
plt.title("ABC bilateral")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(grad_mag_v, cmap="gray")
plt.title("ABC var")
plt.axis("off")
plt.show()
```

## Usage notes

- Use grayscale images as input.
- It is recommended to normalize images to the range `[0, 1]` before calling the methods.
- The parameters `d`, `sC`, and `sS` should be tuned according to noise level and image detail scale.
- For `ABC_var`, small windows such as `3 × 3` or `5 × 5` are a good starting point.

## Technical remarks

- The package implements adaptive edge detectors driven by a local parameter `alpha`.
- In this project, `ABC` stands for **Atangana--Baleanu in the Caputo sense**, not *Artificial Bee Colony*.
- Since the outputs are normalized to the 0--255 range for visualization, they are suitable for visual inspection, but not necessarily for direct comparison of absolute gradient magnitudes across different runs without additional standardization.

## Project metadata

- **Package name:** `myabc`
- **Version:** `0.1.0`
- **Author:** Sergio Renato Rengifo Lozano
- **Required Python version:** `>=3.8`

## License

The source files declare the Apache 2.0 License. It is recommended to verify that the license filename in the repository matches the one referenced in `pyproject.toml`.

## Suggested future improvements

- Add reproducible examples with sample images
- Include automated tests
- Add comparisons against classical edge detectors such as Sobel, Prewitt, and Canny
- Document the mathematical background of the fractional kernel and the local `alpha` estimation
