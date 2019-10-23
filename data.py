from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

import tensorflow as tf

# Mostly from https://github.com/pytorch/examples/tree/master/super_resolution
def download_bsd300(dest="image_data"):
    output_image_dir = join(dest, "BSDS300")

    if not exists(output_image_dir):
        makedirs(output_image_dir)
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
    image= tf.cast(image, tf.float32)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    offset_height = (image_height-crop_size) // 2
    offset_width = (image_width-crop_size) // 2
    original_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)
    downsampled_image = tf.image.resize(original_image, [crop_size // 2, crop_size // 2])
    # convert to 0~1 and change HWC to CHW 
    # (because the network accepts single channel. 
    # See model.py line 12 for better understanding.)
    original_image = tf.transpose(original_image / 255.0, [2, 0, 1])
    downsampled_image = tf.transpose(downsampled_image / 255.0, [2, 0, 1])
    return downsampled_image, original_image

def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train/*.jpeg")
    names = tf.data.Dataset.list_files(train_dir)
    images = names.map(get_image_from_file)
    return images

def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test/*.jpeg")
    names = tf.data.Dataset.list_files(test_dir)
    images = names.map(get_image_from_file)
    return images
    
