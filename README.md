# WORKSHOP-4-DIP

# Workshop 4: Coin Detection and Counting using OpenCV

## Overview
Automated coin detection is a foundational computer vision problem focused on object segmentation, shape recognition, and feature extraction. This workshop covers the identification, boundary delineation, and counting of circular coins against varied backgrounds using digital image processing techniques in OpenCV and Python.

---

## Processing Pipeline

* **1. Grayscale Conversion:** Reduces the input image from 3 color channels to a single intensity channel using `cv2.cvtColor()` to simplify mathematical operations.
* **2. Gaussian Smoothing:** Applies `cv2.GaussianBlur()` with a low-pass filter kernel (e.g., $5 \times 5$ or $9 \times 9$) to eliminate high-frequency specular reflections and surface texture noise on the coins.
* **3. Thresholding / Binarization:** Employs Otsu's automatic thresholding or adaptive thresholding (`cv2.threshold()`) to separate foreground coins from the background.
* **4. Morphological Refinement:** Uses morphological closing (`cv2.MORPH_CLOSE`) to fill internal coin embossments and morphological opening (`cv2.MORPH_OPEN`) to disconnect overlapping coin boundaries.
* **5. Detection & Extraction:**
  * **Approach A (Contour Detection):** Identifies external continuous curves via `cv2.findContours()`, calculates perimeter and area, and filters non-coin artifacts using circularity metrics ($4\pi \times \frac{\text{Area}}{\text{Perimeter}^2}$).
  * **Approach B (Hough Circle Transform):** Detects circular shapes via `cv2.HoughCircles()` by voting in parameter accumulator space ($x, y, r$) using image gradient vectors.
* **6. Visualization & Metrics:** Draws detected boundaries using `cv2.drawContours()` or `cv2.circle()` and overlays the total coin count on the output canvas.

---

## Key OpenCV Functions

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    # Morphological Opening: erosion followed by dilation to remove noise
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    
    # Edge detection and contour mapping
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

def present_results(original_image, processed_image):
    # Convert from BGR (OpenCV) to RGB (Matplotlib)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis('off')

    plt.show()

# --- Main Execution ---
image_path = 'myimage.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from '{image_path}'. Ensure the file is in your current working directory.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)
```

### OUTPUT - 



### RESULT - 
        Thus , the workshop has been implemented successfully.
