import argparse
import tensorflow as tf
from model import ESPCN
from data import get_training_set, get_test_set, get_coco_training_set, get_coco_test_set
import logging
import os
from os.path import join, exists
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
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-save_dir', default='saved_models', type=str)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-use_tpu', type=str2bool, default=False)
parser.add_argument('-save_tflite', type=str2bool, default=False)
parser.add_argument('-tflite_dir', type=str, default='tflite_models')
args = parser.parse_args()
tf.random.set_seed(args.seed)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=200,
    decay_rate=0.99,
    staircase=True)

# TPU objects
tpu_strategy = None

# model & optimizer
model = None
optimizer = None

if args.use_tpu:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_host(cluster_resolver.master())
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    with tpu_strategy.scope():
        model = ESPCN(args.upscale_factor)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
else:
    model = ESPCN(args.upscale_factor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Dataset
train_dataset = None
test_dataset = None
if args.use_tpu:
    with tpu_strategy.scope():
        #train_dataset = get_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
        train_dataset = get_coco_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
        train_dataset = tpu_strategy.experimental_distribute_dataset(train_dataset)
        #test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)
        test_dataset = get_coco_test_set(args.upscale_factor).batch(args.batch_size)
        test_dataset = tpu_strategy.experimental_distribute_dataset(test_dataset)
else:
    train_dataset = get_coco_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
    #train_dataset = get_training_set(args.upscale_factor).shuffle(200).batch(args.batch_size)
    test_dataset = get_coco_test_set(args.upscale_factor).batch(args.batch_size)
    #test_dataset = get_test_set(args.upscale_factor).batch(args.batch_size)

# Train & test steps
train_step = None
test_step = None

if args.use_tpu:
    with tpu_strategy.scope():
        @tf.function
        def train_step_tpu(dist_inputs):
            def step_fn(inputs):
                ds_image, image = inputs
                with tf.GradientTape() as tape:
                    generated_image = model(ds_image)
                    loss_one = tf.reduce_sum(tf.reduce_mean(tf.math.squared_difference(tf.reshape(generated_image, [-1, 256*256*3]), tf.reshape(image, [-1, 256*256*3])), axis=1))
                    loss = loss_one * (1.0 / args.batch_size)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss_one

            per_example_losses = tpu_strategy.experimental_run_v2(
                step_fn, args=(dist_inputs, ))
            mean_loss = tpu_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
            return mean_loss
        train_step = train_step_tpu
            
else:
    @tf.function
    def train_step_normal(inputs):
        ds_image, image = inputs
        with tf.GradientTape() as tape:
            generated_image = model(ds_image)
            loss = tf.reduce_mean(tf.math.squared_difference(generated_image, image))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    train_step = train_step_normal

if args.use_tpu:
    with tpu_strategy.scope():
        @tf.function
        def test_step_tpu(dist_inputs):
            def step_fn(inputs):
                ds_image, image = inputs
                generated_image = model(ds_image)
                loss_one = tf.reduce_sum(tf.reduce_mean(tf.math.squared_difference(generated_image, [-1, 256*256*3]), tf.reshape(image, [-1, 256*256*3]), axis=1))
                return loss_one

            per_example_losses = tpu_strategy.experimental_run_v2(
                step_fn, args=(dist_inputs, ))
            mean_loss = tpu_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
            return mean_loss
        test_step = test_step_tpu
else:
    @tf.function
    def test_step_normal(inputs):
        ds_image, image = inputs
        generated_image = model(ds_image)
        loss = tf.reduce_mean(tf.math.squared_difference(generated_image, image))
        return loss
    test_step = test_step_normal

logging.info('Starting train process. Exp_name: %s' % args.exp_name)
best_model = 0
best_test_loss = 1000000
for epoch in range(args.num_epochs):
    train_loss_sum = 0
    train_cnt = 0
    test_loss_sum = 0
    test_cnt = 0
    if args.use_tpu:
        with tpu_strategy.scope():
            for inputs in train_dataset:
                train_loss_sum += train_step(inputs)
                train_cnt += 1
            for inputs in test_dataset:
                test_loss_sum += test_step(inputs)
                test_cnt += 1
    else:
        for inputs in train_dataset:
            train_loss_sum += train_step(inputs)
            train_cnt += 1
        for inputs in test_dataset:
            test_loss_sum += test_step(inputs)
            test_cnt += 1

    if best_test_loss > (test_loss_sum / test_cnt):
        best_model = epoch
        best_test_loss = (test_loss_sum / test_cnt)

    save_path = join(args.save_dir, args.exp_name, str(epoch))
    if not exists(save_path):
        os.makedirs(save_path)
    tf.saved_model.save(model, save_path)
    logging.info('epoch: %d, train_loss: %f, test_loss: %f' % (epoch+1, train_loss_sum / train_cnt, test_loss_sum / test_cnt))
    if args.save_tflite:
        tflite_file = join(args.tflite_dir, args.exp_name, str(epoch)+'_256.tflite')
        converter = tf.lite.TFLiteConverter.from_concrete_functions([tf.function(model.call, input_signature=(tf.TensorSpec(shape=(None, 256, 256, 3)), ))])
        tflite_model = converter.convert()
        open(tflite_file, 'wb').write(tflite_model)
        tflite_file = join(args.tflite_dir, args.exp_name, str(epoch)+'_512.tflite')
        converter = tf.lite.TFLiteConverter.from_concrete_functions([tf.function(model.call, input_signature=(tf.TensorSpec(shape=(None, 256, 256, 3)), ))])
        tflite_model = converter.convert()
        open(tflite_file, 'wb').write(tflite_model)
        

print('best model: %d' % best_model)
