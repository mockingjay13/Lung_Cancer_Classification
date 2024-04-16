import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

# Load lung CT scan image
def load_image(image_path):
    img = imread(image_path, as_gray=True)
    return img

# Preprocess image (e.g., normalization)
def preprocess_image(image):
    # Perform any necessary preprocessing steps here
    # For example, normalization or resizing
    return image

# Perform K-means clustering for tumor segmentation
def segment_tumor(image, num_clusters=2):
    # Apply thresholding (e.g., Otsu's method) to preprocess the image
    threshold = threshold_otsu(image)
    binary_image = image > threshold
    
    # Reshape the image for clustering
    flat_image = binary_image.flatten().reshape(-1, 1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flat_image)
    segmented_image = kmeans.labels_.reshape(binary_image.shape)
    return segmented_image

# Visualize segmented tumor regions
def visualize_segmentation(image, segmented_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='viridis')
    plt.title('Segmented Tumor Regions')
    plt.show()

# Main function
def main():
    # Load and preprocess lung CT scan image
    image_path = 'test.png'
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    
    # Perform tumor segmentation using K-means clustering
    segmented_image = segment_tumor(preprocessed_image)
    
    # Visualize the segmented tumor regions
    visualize_segmentation(image, segmented_image)

if __name__ == '__main__':
    main()
