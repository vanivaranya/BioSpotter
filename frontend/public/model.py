# import os
# import json
# import tempfile
# from kaggle.api.kaggle_api_extended import KaggleApi
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import matplotlib.pyplot as plt

# # Set the path to the kaggle.json file
# kaggle_config_path = r'C:\Users\hp\.kaggle\kaggle.json'

# # Ensure the directory and file exist
# if not os.path.exists(kaggle_config_path):
#     raise FileNotFoundError(f"Could not find {kaggle_config_path}")

# # Load the Kaggle API credentials
# with open(kaggle_config_path, 'r') as f:
#     kaggle_creds = json.load(f)

# os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
# os.environ['KAGGLE_KEY'] = kaggle_creds['key']

# api = KaggleApi()
# api.authenticate()

# # Download the dataset
# dataset = 'iamsouravbanerjee/animal-image-dataset-90-different-animals'
# with tempfile.TemporaryDirectory() as temp_dir:
#     api.dataset_download_files(dataset, path=temp_dir, unzip=True)
#     dataset_path = os.path.join(temp_dir, 'animals', 'animals')

#     # Initialize data dictionary
#     data = {"imgpath": [], "labels": []}

#     # Walk through the directory and gather image paths and labels
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.endswith(".jpg"):
#                 label = root.split(os.sep)[-1]
#                 imgpath = os.path.join(root, file)
#                 data["imgpath"].append(imgpath)
#                 data["labels"].append(label)

#     df = pd.DataFrame(data)
#     print(f"Total images: {len(df)}")
    
#     # Convert labels to numbers
#     df['encoded_labels'] = pd.factorize(df['labels'])[0]
#     num_classes = len(df['labels'].unique())

#     # Split the dataset into train, validation, and test sets
#     train_df, temp_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=124)
#     valid_df, test_df = train_test_split(temp_df, train_size=0.7, shuffle=True, random_state=124)
#     train_df = train_df.reset_index(drop=True)
#     valid_df = valid_df.reset_index(drop=True)
#     test_df = test_df.reset_index(drop=True)

#     # Define data generators
#     BATCH_SIZE = 15
#     IMAGE_SIZE = (224, 224)

#     generator = ImageDataGenerator(
#         preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )

#     train_images = generator.flow_from_dataframe(
#         dataframe=train_df,
#         x_col='imgpath',
#         y_col='labels',
#         target_size=IMAGE_SIZE,
#         color_mode='rgb',
#         class_mode='categorical',
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         seed=42
#     )

#     val_images = generator.flow_from_dataframe(
#         dataframe=valid_df,
#         x_col='imgpath',
#         y_col='labels',
#         target_size=IMAGE_SIZE,
#         color_mode='rgb',
#         class_mode='categorical',
#         batch_size=BATCH_SIZE,
#         shuffle=False
#     )

#     test_images = generator.flow_from_dataframe(
#         dataframe=test_df,
#         x_col='imgpath',
#         y_col='labels',
#         target_size=IMAGE_SIZE,
#         color_mode='rgb',
#         class_mode='categorical',
#         batch_size=BATCH_SIZE,
#         shuffle=False
#     )

#     # Build the model
#     inputs = layers.Input(shape=(224, 224, 3), name='inputLayer')
#     x = layers.Rescaling(1./255)(inputs)

#     x = layers.Conv2D(32, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D(2, 2)(x)

#     x = layers.Flatten()(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=inputs, outputs=x)

#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     # Train the model
#     history = model.fit(
#         train_images,
#         epochs=1,
#         validation_data=val_images,
#         verbose=1
#     )

#     # Evaluate the model on test data
#     loss, accuracy = model.evaluate(test_images, verbose=0)
#     print(f'Test Loss: {loss:.4f}')
#     print(f'Test Accuracy: {accuracy:.4f}')

#     # Save the model
#     model.save("animal_classifier_model.keras")

# # Load the saved model
# model = tf.keras.models.load_model("animal_classifier_model.keras")

# # Function to load and preprocess an image
# def load_and_preprocess_image(img_path, target_size):
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Preprocess for EfficientNet
#     return img_array

# # Function to display an image with a prediction
# def display_prediction(img_path):
#     img_array = load_and_preprocess_image(img_path, IMAGE_SIZE)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     # Get the class label from the encoded labels
#     label_map = {v: k for k, v in pd.Series(df['encoded_labels'].values, df['labels']).items()}
#     predicted_label = label_map[predicted_class]
    
#     # Display the image and prediction
#     img = image.load_img(img_path)
#     plt.imshow(img)
#     plt.title(f"Prediction: {predicted_label}")
#     plt.axis('off')
#     plt.show()

# # Example usage
# img_path = r"C:\Users\hp\Desktop\WebDev\BioSpotter\dog.jpg"  # Replace with the path to your image
# display_prediction(img_path)

import os
import json
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the kaggle.json file
kaggle_config_path = r'C:\Users\hp\.kaggle\kaggle.json'

# Ensure the directory and file exist
if not os.path.exists(kaggle_config_path):
    raise FileNotFoundError(f"Could not find {kaggle_config_path}")

# Load the Kaggle API credentials
with open(kaggle_config_path, 'r') as f:
    kaggle_creds = json.load(f)

os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
os.environ['KAGGLE_KEY'] = kaggle_creds['key']

api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'iamsouravbanerjee/animal-image-dataset-90-different-animals'
with tempfile.TemporaryDirectory() as temp_dir:
    api.dataset_download_files(dataset, path=temp_dir, unzip=True)
    dataset_path = os.path.join(temp_dir, 'animals', 'animals')

    # Initialize data dictionary
    data = {"imgpath": [], "labels": []}

    # Walk through the directory and gather image paths and labels
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                label = root.split(os.sep)[-1]
                imgpath = os.path.join(root, file)
                data["imgpath"].append(imgpath)
                data["labels"].append(label)

    df = pd.DataFrame(data)
    print(f"Total images: {len(df)}")

    # Convert labels to numbers
    df['encoded_labels'] = pd.factorize(df['labels'])[0]
    num_classes = len(df['labels'].unique())

    # Save the class label mapping to a JSON file
    label_map = {str(v): k for k, v in enumerate(df['labels'].unique())}
    with open('class_labels.json', 'w') as f:
        json.dump(label_map, f)

    # Split the dataset into train, validation, and test sets
    train_df, temp_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=124)
    valid_df, test_df = train_test_split(temp_df, train_size=0.7, shuffle=True, random_state=124)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Define data generators
    BATCH_SIZE = 15
    IMAGE_SIZE = (224, 224)

    generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_images = generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='imgpath',
        y_col='labels',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_images = generator.flow_from_dataframe(
        dataframe=valid_df,
        x_col='imgpath',
        y_col='labels',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_images = generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='imgpath',
        y_col='labels',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Build the model
    inputs = layers.Input(shape=(224, 224, 3), name='inputLayer')
    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_images,
        epochs=1,
        validation_data=val_images,
        verbose=1
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_images, verbose=0)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save the model
    model.save("animal_classifier_model.keras")

# Load the saved model
model = tf.keras.models.load_model("animal_classifier_model.keras")

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Preprocess for EfficientNet
    return img_array

# Function to display an image with a prediction
def display_prediction(img_path):
    img_array = load_and_preprocess_image(img_path, IMAGE_SIZE)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Load the class label from the encoded labels
    with open('class_labels.json', 'r') as f:
        label_map = json.load(f)
    predicted_label = label_map[str(predicted_class)]
    
    # Display the image and prediction
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

# # Example usage
# img_path = r"C:\Users\hp\Desktop\WebDev\BioSpotter\dog.jpg"  # Replace with the path to your image
# display_prediction(img_path)
