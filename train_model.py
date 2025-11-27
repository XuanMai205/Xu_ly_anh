import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


NUM_CLASSES = 10


def build_cnn_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print('---Bat dau huan luyen mo hinh CNN---')

   # Tải dữ liệu MNIST (chữ số 0-9)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # so lop chu cai

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    model = build_cnn_model(NUM_CLASSES)
    print('Bat dau huan luyen...')
    model.fit(x_train, y_train, epochs=10, batch_size=64,
              validation_data=(x_test, y_test))

    model.save('handwritten_recognizer_model.h5')
    print('Mo hinh da duoc luu vao handwritten_recognizer_model.h5')
