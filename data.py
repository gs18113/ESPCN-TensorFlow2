from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

import tensorflow as tf
from tensorflow.data import Dataset

# Mostly from https://github.com/pytorch/examples/tree/master/super_resolution
def download_bsd300(dest="image_data"):
    output_image_dir = join(dest, "BSDS300")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir

def get_image_from_file(filename, crop_size=256):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    offset_height = (image_height-crop_size) // 2
    offset_width = (image_width-crop_size) // 2
    original_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)
    downsampled_image = tf.image.resize(original_image, [crop_size // 2, crop_size // 2])
    return original_image, downsampled_image

def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    names = Dataset.list_files(train_dir)
    images = names.map(get_image_from_file)
    return images

def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    names = Dataset.list_files(test_dir)
    images = names.map(get_image_from_file)
    return images
    