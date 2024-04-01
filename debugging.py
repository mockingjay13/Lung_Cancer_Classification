from keras.applications import VGG16

# Load VGG16 model without including the fully-connected layers at the top
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Print the input shape expected by the first layer
print("Input shape expected by the model:", base_model.input_shape)