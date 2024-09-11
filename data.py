import tensorflow as tf
import pandas as pd


BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, dtype='float32')
    image.set_shape((64, 64, 1))
    label[0].set_shape([])
    label[1].set_shape([])
    labels = {"first_number": label[0], "second_number": label[1]}
    return image, labels


def get_dataset():
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    train_images = 'dataset/train_images/' + train_df.iloc[:, 0].values
    test_images = 'dataset/test_images/' + test_df.iloc[:, 0].values
    train_labels = train_df.iloc[:, 1:].values
    test_labels = test_df.iloc[:, 1:].values
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    )

    train_dataset = (
        train_dataset.shuffle(buffer_size=len(train_labels))
        .map(read_image)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)
    )

    test_dataset = (
        test_dataset.map(read_image)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    return train_dataset, test_dataset