from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    InputLayer,
    Dropout,
    Flatten,
    Reshape,
    LeakyReLU,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from utils import CustomReshapeLayer


def yolo_model():
    # We will follow the leaky relu activation as mentioned in the paper
    leakyR = LeakyReLU(alpha=0.1)

    # Stack 1
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            input_shape=(448, 448, 3),
            padding="same",
            activation=leakyR,
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # Stack2
    model.add(
        Conv2D(filters=192, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # Stack3
    model.add(
        Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # Stack4
    model.add(
        Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # Stack5
    model.add(
        Conv2D(filters=512, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=512, kernel_size=(1, 1), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(
        Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation=leakyR)
    )
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same"))
    # Stack6
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=leakyR))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=leakyR))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1470, activation="sigmoid"))
    model.add(CustomReshapeLayer(target_shape=(7, 7, 30)))
    # model.summary()
    return model
