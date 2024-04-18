import keras
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers
from keras.applications import ResNet50,VGG16,ResNet101, VGG19, DenseNet201, EfficientNetB4, MobileNetV2
from keras.applications import vgg16 
from keras import Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import os
import cv2
import zipfile
import splitfolders


##############################################################################################################################################

# Data Pre-processing

##############################################################################################################################################

# unzip to extract raw data files
rawData_directory = 'Raw_Data'
categories = ['adenocarcinoma', 'largecellcarcinoma', 'normal', 'squamouscellcarcinoma']
processed_directory = 'Processed_Data'

def unzip_folder(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

# Specify the path to the zipped file and the output directory
zipFile_path = 'Raw_Data_zipped.zip'
unzip_directory = 'Raw_Data'  

# Create the output directory if it doesn't exist
if not os.path.exists(unzip_directory):
    os.makedirs(unzip_directory)

# Unzip the folder
unzip_folder(zipFile_path, unzip_directory)

for category in categories:
    path = os.path.join(rawData_directory, category)
    for image in os.listdir(path):
        image_Path = os.path.join(path, image)
        readImage = cv2.imread(image_Path, 0)
        equalizedImage = cv2.equalizeHist(readImage)
        e, segmentedImage = cv2.threshold(equalizedImage, 128, 255, cv2.THRESH_TOZERO)
        if category == 'normal':
            imageFinal = image_Path.replace('Raw_Data\\normal', 'Processed_Data\\normal')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'adenocarcinoma':
            imageFinal = image_Path.replace('Raw_Data\\adenocarcinoma', 'Processed_Data\\adenocarcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'largecellcarcinoma':
            imageFinal = image_Path.replace('Raw_Data\\largecellcarcinoma', 'Processed_Data\\largecellcarcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'squamouscellcarcinoma':
            imageFinal = image_Path.replace('Raw_Data\\squamouscellcarcinoma', 'Processed_Data\\squamouscellcarcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
print("Successfully created processed data directory at:", processed_directory)


###############################################################################################################################################

# Splitting the processed images

###############################################################################################################################################

split_processed_directory = './Processed_Data_Split'

splitfolders.ratio(processed_directory, output=split_processed_directory, seed=6942, ratio=(0.7, 0.1, 0.2)) 


###############################################################################################################################################

# Neural networks

###############################################################################################################################################

N_CLASSES = 4
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(dtype='float32', rescale= 1.0/255.0)

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1.0/255.0)

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)

train_DataSet  = train_datagen.flow_from_directory(directory = 'Processed_Data_Split/train',
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (224,224),
                                                   class_mode = 'categorical')

validate_DataSet = valid_datagen.flow_from_directory(directory = 'Processed_Data_Split/val',
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (224,224),
                                                    class_mode = 'categorical')

test_DataSet = test_datagen.flow_from_directory(directory = 'Processed_Data_Split/test',
                                                 batch_size = BATCH_SIZE,
                                                 target_size = (224,224),
                                                 class_mode = 'categorical')


##############################################################################################################################################

# VGG16

##############################################################################################################################################

base_model = VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# Define input layer
input_layer = Input(shape=(224, 224, 3))  # shape_of_input_tensor should match the shape of input to `base_model`

# Pass the input to base_model
base_model_output = base_model(input_layer)

# Flatten the output of base_model
flattened_output = Flatten()(base_model_output)

# Add BatchNormalization layer
normalized_output = BatchNormalization()(flattened_output)

# Add Dense layer for classification
output_layer = Dense(N_CLASSES, activation='softmax')(normalized_output)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)


#model = Sequential()
#model.add(base_model)
#model.add(Flatten())
#model.add(BatchNormalization())
#model.add(Dense(N_CLASSES, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

checkpointer = ModelCheckpoint(filepath='vgg16_CheckPoint.keras',
                            monitor='val_loss', verbose = 1,
                            save_best_only=True)
early_stopping = EarlyStopping(verbose=1, patience=15)

model_history = model.fit(train_DataSet,
                    steps_per_epoch = 20,
                    epochs = 100,
                    verbose = 1,
                    validation_data = validate_DataSet,
                    callbacks = [checkpointer, early_stopping])

model_scores = model.evaluate(test_DataSet)

#plt.plot(model_history.history['acc'], label = 'train',)
#plt.plot(model_history.history['val_acc'], label = 'val')

#plt.legend(loc = 'right')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.show()

# Plotting accuracy curves
#plt.plot(model_history.history['acc'], label='Training Accuracy', color='blue')
#plt.plot(model_history.history['val_acc'], label='Validation Accuracy', color='orange')

# Adding title and labels
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Smoothing the curves
smooth_window = 5
smoothed_train_acc = [sum(model_history.history['acc'][i:i+smooth_window])/smooth_window for i in range(len(model_history.history['acc'])-smooth_window)]
smoothed_val_acc = [sum(model_history.history['val_acc'][i:i+smooth_window])/smooth_window for i in range(len(model_history.history['val_acc'])-smooth_window)]
#plt.plot(smoothed_train_acc, label='Smoothed Training Accuracy', color='darkblue', linestyle='--')
#plt.plot(smoothed_val_acc, label='Smoothed Validation Accuracy', color='darkorange', linestyle='--')
plt.plot(smoothed_train_acc, label='Smoothed Training Accuracy', color='darkblue')
plt.plot(smoothed_val_acc, label='Smoothed Validation Accuracy', color='darkorange')

# Adding legend
plt.legend(loc='lower right')

# Displaying the plot
plt.show()


# Predict labels for test dataset
y_pred = model.predict(test_DataSet)
y_pred_classes = np.argmax(y_pred, axis=1)
true_classes = test_DataSet.classes

# Get class labels
class_labels = list(test_DataSet.class_indices.keys())

# Print classification report
print("Classification Report:")
print(classification_report(true_classes, y_pred_classes, target_names=class_labels))

# Plot confusion matrix
cm = confusion_matrix(true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Fill the confusion matrix with numbers
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

# Show the plot
plt.tight_layout()
plt.show()


# Save the model
model.save('lung_cancer_trained_model.keras')

accuracy_val = model_scores[1]
print('Accuracy: ', accuracy_val)

with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy_val))