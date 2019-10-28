# for colab
# from google.colab import auth

# def authenticate():
#   auth.authenticate_user()

# authenticate()

from google_storage_utils import GCS
from private_constants import PrivateConstants
from utils import Utils
from constants import Constants
from style_transfer_trainer import StyleTransferTrainer
from style_transfer_tester import StyleTransferTester
from vgg19 import VGG19

import os
import tensorflow as tf
import requests
from io import BytesIO
import urllib.request
from skimage import io
from skimage import transform
import numpy as np
import time


def getResources():
    if not os.path.exists("/tmp/pre_trained_model"):
        os.makedirs("/tmp/pre_trained_model")
    GCS.download_file_from_gcs(
        bucket_name=PrivateConstants.BUCKET_NAME,
        from_file=PrivateConstants.MODEL_PATH + "/vgg-19/vgg19.mat",
        to_file="/tmp/pre_trained_model/vgg19.mat")
    if not os.path.exists("/tmp/style"):
        os.makedirs("/tmp/style")
    GCS.download_files_from_gcs(
        PrivateConstants.BUCKET_NAME,
        PrivateConstants.DATA_PATH + "/style/",
        "/tmp/style")
    if not os.path.exists("/tmp/myDB"):
        os.makedirs("/tmp/myDB")
    GCS.download_files_from_gcs(
        PrivateConstants.BUCKET_NAME,
        PrivateConstants.DATA_PATH + "/iphone-db/",
        "/tmp/myDB")


def getOriginalDB():
    if not os.path.exists("/tmp/train2014"):
        os.makedirs("/tmp/train2014")
    # !gsutil -m cp gs://{PrivateConstants.BUCKET_NAME}/{PrivateConstants.DATA_PATH}/train2014/ /tmp/train2014



def train(train_db_path="/tmp/myDB", num_epochs=2, batch_size=4, check_point_every=100):
    # Training
    vgg_model = '/tmp/pre_trained_model'
    trainDB_path = train_db_path
    style = '/tmp/style/udnie.jpg'  # swap with my path
    style_name = "udnie"
    content_layers_weights = [1.0]
    style_layer_weights = [.2, .2, .2, .2, .2]
    content_layers = ['relu4_2']
    content_layer_weights = [1.0]
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    style_layer_weights = [.2, .2, .2, .2, .2]
    content_weight = 7.5e0
    style_weight = 5e2
    tv_weight = 2e2
    num_epochs = num_epochs
    batch_size = batch_size
    learn_rate = 1e-3
    output = '/tmp/models'
    checkpoint_every = check_point_every
    test = None
    max_size = None
    model_file_path = vgg_model + '/' + Constants.MODEL_FILE_NAME

    vgg_net = VGG19(model_file_path)
    content_images = Utils.get_files(trainDB_path)
    style_image = Utils.load_image(style)

    CONTENT_LAYERS = {}
    for layer, weight in zip(content_layers, content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    STYLE_LAYERS = {}
    for layer, weight in zip(style_layers, style_layer_weights):
        STYLE_LAYERS[layer] = weight

    tf.reset_default_graph()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    trainer = StyleTransferTrainer(session=sess,
                                   content_layer_ids=CONTENT_LAYERS,
                                   style_layer_ids=STYLE_LAYERS,
                                   content_images=content_images,
                                   style_image=Utils.add_one_dim(style_image),
                                   net=vgg_net,
                                   num_epochs=num_epochs,
                                   batch_size=batch_size,
                                   content_weight=content_weight,
                                   style_weight=style_weight,
                                   tv_weight=tv_weight,
                                   learn_rate=learn_rate,
                                   save_path=output,
                                   check_period=checkpoint_every,
                                   test_image=test,
                                   max_size=max_size,
                                   style_name=style_name)

    trainer.train()

    sess.close()


def test(iteration="23000", \
         url="https://raw.githubusercontent.com/hwalsuklee/tensorflow-fast-style-transfer/master/content/chicago.jpg", \
         feed_shape=None):
    # load content image
    max_size = None

    tf.reset_default_graph()

    dest_path = "/tmp/test/temp.jpg"
    model_path = "/tmp/models/final.ckpt-" + iteration
    res_path = "/tmp/test/result.jpg"

    if not os.path.exists("/tmp/test/"):
        os.makedirs("/tmp/test/")

    with urllib.request.urlopen(url) as url:
        with open('/tmp/test/temp.jpg', 'wb') as f:
            f.write(url.read())

    img = io.imread(dest_path)
    io.imshow(img)
    content_image = Utils.load_image(dest_path, shape=max_size)

    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True  # to deal with large image
    sess = tf.Session(config=soft_config)

    resize_start = time.time()
    if feed_shape:
        pre_size = content_image.shape
        content_image = transform.resize(content_image, feed_shape, preserve_range=True)

    # build the graph
    construction_start = time.time()
    transformer = StyleTransferTester(session=sess,
                                      model_path=model_path,
                                      content_image=content_image)
    # execute the graph
    start_time = time.time()
    output_image = transformer.test()
    end_time = time.time()
    print(f"out max: {output_image.max()}")
    print(f"out min: {output_image.min()}")
    print(f"out mean: {output_image.mean()}")
    print(f"output_shape: {output_image.shape}")
    print(f"output_type: {type(output_image)}")
    # report execution time
    shape = content_image.shape  # (batch, width, height, channel)
    print('Execution time for a %d x %d image : %f msec' % (
    shape[0], shape[1], 1000. * float(end_time - start_time) / 60))
    clipped = np.clip(output_image, 0.0, 255.0)
    print(f"clipped max: {clipped.max()}")
    print(f"clipped min: {clipped.min()}")
    print(f"clipped mean: {clipped.mean()}")

    if feed_shape:
        # res_downsized = downsized # I don't know why I had this there, or if it ran, but it probably would have broken
        clipped = transform.resize(clipped, pre_size, preserve_range=True)

    resize_end = time.time()
    print('Total time with resize %f msec' % (
                1000. * float(resize_end - resize_start - (start_time - construction_start)) / 60))
    io.imshow(clipped.astype(np.uint8))

    # save result


#     Utils.save_image(output_image, res_path)


if __name__ == "__main__":
    train()
