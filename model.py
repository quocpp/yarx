from keras.applications.densenet import DenseNet121
from keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
import keras


def create_model(labels):
    loss_fn = keras.losses.CategoricalCrossentropy()
    model = DenseNet121(weights='densenet.hdf5', include_top=False)
    model = Model(inputs=model.input, outputs=Dense(len(labels), activation="sigmoid")(GlobalAveragePooling2D()(model.output)))
    model.compile(optimizer='adam', loss=loss_fn)
    return model
