import keras
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
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

# Save the model
model.save('lung_cancer_trained_model.keras')

accuracy_val = model_scores[1]
print('Accuracy: ', accuracy_val)

with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy_val))