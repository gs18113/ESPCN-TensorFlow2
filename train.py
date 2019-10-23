import argparse
import tensorflow as tf
from model import ESPCN
from data import get_training_set, get_test_set
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-upscale_factor', default=2, type=int)
parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-seed', default=123, type=int)
args = parser.parse_args()
tf.random.set_seed(args.seed)
model = ESPCN(args.upscale_factor)

# Dataset
train_dataset = get_training_set(args.upscale_factor).batch(args.batch_size)
test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)

# Loss & optimizer
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Keras metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


def train_step(ds_image, image):
    with tf.GradientTape() as tape:
        generated_image = model(ds_image)
        loss = loss_object(generated_image, image)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


def test_step(ds_image, image):
    generated_image = model(ds_image)
    t_loss = loss_object(generated_image, image)
    test_loss(t_loss)

for epoch in range(args.num_epochs):
    for ds_image, image in train_dataset:
        train_step(ds_image, image)
    for test_ds_image, test_image in test_dataset:
        test_step(test_ds_image, test_image)
    logging.info('epoch: %d, train_loss: %f, test_loss: %f' % (epoch+1, train_loss.result(), test_loss.result()))
