import argparse
import tensorflow as tf



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-upscale_factor', default=2, type=int)
    parser.add_argument('-num_epochs', default=100, type=int)