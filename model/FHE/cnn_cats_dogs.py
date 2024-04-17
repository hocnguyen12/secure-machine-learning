import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Activation
from activations import SquareActivation

import matplotlib.pyplot as plt
import h5py

train_cats_folder = 'training_set/cats'
train_dogs_folder = 'training_set/dogs'
test_cats_folder = 'test_set/cats'
test_dogs_folder = 'test_set/dogs'

batch_size = 128
epochs = 50

def load_data(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            try:
                image = Image.open(image_path).convert('RGB').resize((50, 50))
                images.append(np.array(image) / 255.0)
                if 'cats' in folder:
                    labels.append(0)
                elif 'dogs' in folder:
                    labels.append(1)
            except IOError:
                print(f"Cannot load image: {image_path}")
    return images, labels

print("Loading training data...")
# Load training data
train_cat_images, train_cat_labels = load_data(train_cats_folder)
train_dog_images, train_dog_labels = load_data(train_dogs_folder)

print("Loading validation data...")
# Load test data
test_cat_images, test_cat_labels = load_data(test_cats_folder)
test_dog_images, test_dog_labels = load_data(test_dogs_folder)

print("Data loaded")

# Combine training data
train_images = np.concatenate([train_cat_images, train_dog_images], axis=0)
train_labels = np.concatenate([train_cat_labels, train_dog_labels], axis=0)

# Combine test data
test_images = np.concatenate([test_cat_images, test_dog_images], axis=0)
test_labels = np.concatenate([test_cat_labels, test_dog_labels], axis=0)

num_classes = 2

train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print("train and test labels done")

input_shape = x_train[0].shape

# HOMOMORPHIC ENCRYPTION FRIENDLY MODEL
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=3, padding='valid', input_shape=input_shape))
model.add(Flatten())
model.add(SquareActivation())
model.add(Dense(100))
model.add(SquareActivation())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Training
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save model
model_folder = 'data/cats_dogs'
model.save_weights(os.path.join(model_folder, 'cats_dogs_model.h5'))

# Save model as JSON
model_json = model.to_json()
with open(os.path.join(model_folder, 'cats_dogs_model.json'), 'w') as json_file:
    json_file.write(model_json)

def save_data_set(x, y, data_type, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f'x_{data_type}.h5')
    print(f"Saving x_{data_type} of shape {x.shape} in {file_path}")
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset(f'x_{data_type}', data=x)

    file_path = os.path.join(folder_path, f'y_{data_type}.h5')
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset(f'y_{data_type}', data=y)

# Save data set to encrypt later
save_data_set(test_images, test_labels, data_type='test', folder_path=model_folder)
save_data_set(x_train, y_train, data_type='train', folder_path=model_folder)
save_data_set(x_val, y_val, data_type='val', folder_path=model_folder)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy over epochs for HElayer model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('data/cats_dogs/training_validation_accuracy.png')
plt.show()