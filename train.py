import argparse
import tensorflow as tf
from model import ESPCN
from data import get_training_set, get_test_set
import logging
from os.path import join
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-upscale_factor', default=2, type=int)
parser.add_argument('-num_epochs', default=100, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-lr', default=0.01, type=float)
parser.add_argument('-save_dir', default='saved_models', type=str)
parser.add_argument('-use_tpu', type=str2bool, nargs='?', default=False)
args = parser.parse_args()
tf.random.set_seed(args.seed)


# TPU objects
tpu_strategy = None

model = None
if args.use_tpu:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_host(cluster_resolver.master())
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    with tpu_strategy.scope():
        model = ESPCN(args.upscale_factor)
else:
    model = ESPCN(args.upscale_factor)

# Loss & optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=400,
    decay_rate=0.99,
    staircase=True)
optimizer = None
if args.use_tpu:
    with tpu_strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Dataset
train_dataset = None
test_dataset = None
if args.use_tpu:
    with tpu_strategy.scope():
        train_dataset = get_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
        test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)
else:
    train_dataset = get_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
    test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)

# Train & test steps
train_step = None
test_step = None

if args.use_tpu:
    @tf.function
    def train_step_tpu(dist_inputs):
        def step_fn(inputs):
            ds_image, image = inputs
            with tf.GradientTape() as tape:
                generated_image = model(ds_image)
                loss_one = tf.reduce_sum(tf.reduce_mean(tf.reshape(tf.math.squared_difference(generated_image, image), [-1, 256*256]), 1))
                loss = loss_one * (1.0 / args.batch_size)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss_one

        per_example_losses = tpu_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs, ))
        mean_loss = tpu_strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_example_losses)
        return mean_loss
    train_step = train_step_tpu
            
else:
    @tf.function
    def train_step_normal(ds_image, image):
        with tf.GradientTape() as tape:
            generated_image = model(ds_image)
            loss = tf.reduce_sum(tf.reduce_mean(tf.reshape(tf.math.squared_difference(generated_image, image), [-1, 256*256]), 1))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    train_step = train_step_normal

if args.use_tpu:
    @tf.function
    def test_step_tpu(dist_inputs):
        def step_fn(inputs):
            ds_image, image = inputs
            generated_image = model(ds_image)
            loss_one = tf.reduce_sum(tf.reduce_mean(tf.reshape(tf.math.squared_difference(generated_image, image), [-1, 256*256]), 1))
            return loss_one

        per_example_losses = tpu_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs, ))
        mean_loss = tpu_strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_example_losses)
        return mean_loss
    test_step = test_step_tpu
else:
    @tf.function
    def test_step_normal(ds_image, image):
        generated_image = model(ds_image)
        loss = tf.reduce_sum(tf.reduce_mean(tf.reshape(tf.math.squared_difference(generated_image, image), [-1, 256*256]), 1))
        return loss
    test_step = test_step_normal

for epoch in range(args.num_epochs):
    train_loss_sum = 0
    train_cnt = 0
    if args.use_tpu:
        with tpu_strategy.scope():
            for inputs in train_dataset:
                train_loss_sum += train_step(inputs)
                train_cnt += 1
    else:
        for ds_image, image in train_dataset:
            train_loss_sum += train_step(ds_image, image)
            train_cnt += 1

    test_loss_sum = 0
    test_cnt = 0
    if args.use_tpu:
        with tpu_strategy.scope():
            for inputs in test_dataset:
                test_loss_sum += test_step(inputs)
                test_cnt += 1
    else:
        for test_ds_image, test_image in test_dataset:
            test_loss_sum += test_step(test_ds_image, test_image)
            test_cnt += 1

    save_path = join(args.save_dir, str(epoch))
    tf.saved_model.save(model, save_path)
    logging.info('epoch: %d, train_loss: %f, test_loss: %f' % (epoch+1, train_loss_sum / train_cnt, test_loss_sum / test_cnt))
