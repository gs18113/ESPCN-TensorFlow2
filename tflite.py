from tensorflow import lite
import argparse
import logging
import os 
from os.path import join, exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-model_epoch', type=int, required=True)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-saved_dir', default='saved_models', type=str)
parser.add_argument('-tflite_dir', default='tflite_model', type=str)
args = parser.parse_args()

saved_path = join(join(args.saved_dir, args.exp_name), str(args.model_epoch))
logging.info('Creating converter from saved path %s...' % saved_path)
assert exists(saved_path)
converter = lite.TFLiteConverter.from_saved_model(saved_path)
tflite_model = converter.convert()
if not exists(args.tflite_dir):
    os.makedirs(args.tflite_dir)

open(join(args.tflite_dir, 'converted_model_'+args.exp_name+'_epoch_'+str(args.model_epoch)+'.tflite')).write(tflite_model)
logging.info('Converting successful!')