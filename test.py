import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

# Load lung CT scan image
def load_image(image_path):
    return imread(image_path, as_gray=True)

# Preprocess image (e.g., normalization)
def preprocess_image(image):
    # Perform any necessary preprocessing steps here
    return image

# Perform K-means clustering for tumor segmentation
# Perform thresholding for tumor segmentation
def segment_tumor(image, threshold_value):
    # Apply thresholding
    binary_image = np.where(image > threshold_value, 1, 0).astype(np.uint8)
    return binary_image

# Visualize segmented tumor regions
def visualize_segmentation(image, segmented_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Tumor Regions')
    plt.show()

# Main function
def main():
    # Load and preprocess lung CT scan image
    image_path = 'test2.png'
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    
    # Perform tumor segmentation using thresholding
    segmented_image = segment_tumor(preprocessed_image, threshold_value=0.47)
    
    # Visualize the segmented tumor regions
    visualize_segmentation(image, segmented_image)

if __name__ == '__main__':
    main()
