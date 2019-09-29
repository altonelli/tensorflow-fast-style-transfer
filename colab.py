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
from vgg19 import VGG19

import os
import tensorflow as tf


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



def train():
    # Training
    vgg_model = '/tmp/pre_trained_model'
    trainDB_path = '/tmp/myDB'
    style = '/tmp/style/udnie.jpg'  # swap with my path
    content_layers_weights = [1.0]
    style_layer_weights = [.2, .2, .2, .2, .2]
    content_layers = ['relu4_2']
    content_layer_weights = [1.0]
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    style_layer_weights = [.2, .2, .2, .2, .2]
    content_weight = 7.5e0
    style_weight = 5e2
    tv_weight = 2e2
    num_epochs = 2
    batch_size = 4
    learn_rate = 1e-3
    output = '/tmp/models'
    checkpoint_every = 1000
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
                                   max_size=max_size)

    trainer.train()

    sess.close()


if __name__ == "__main__":
    train()
