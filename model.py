import tensorflow as tf
from tensorflow import keras

class MyModel(keras.Model):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims
        self.feature_extractor = keras.Sequential([
            keras.Input(shape=(in_dims)),
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu')
        ])
        self.classifier1 = keras.layers.Dense(out_dims, activation='softmax', name='first_number')
        self.classifier2 = keras.layers.Dense(out_dims, activation='softmax', name='second_number')
        
    
    def call(self, x, training=False):
        x = self.feature_extractor(x)
        output1 = self.classifier1(x)
        output2 = self.classifier2(x)
        return [output1, output2]
    