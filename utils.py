import numpy as np
import os
from skimage import io, transform
from datetime import datetime


class Utils:
    """Helper-functions to load MSCOCO DB"""


    # borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
    @staticmethod
    def get_img(src, img_size=False):
        img = io.imread(src)
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))
        if img_size != False:
            img = transform.resize(img, img_size)
        return img

    @staticmethod
    def get_files(img_dir):
        files = Utils.list_files(img_dir)
        return list(map(lambda x: os.path.join(img_dir, x), files))

    @staticmethod
    def list_files(in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        return files


    """Helper-functions for image manipulation"""


    # borrowed from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb

    # This function loads an image and returns it as a numpy array of floating-points.
    # The image can be automatically resized so the largest of the height or width equals max_size.
    # or resized to the given shape
    @staticmethod
    def load_image(filename, shape=None, max_size=None):
        print(os.path.isfile(filename))
        image = io.imread(filename)
        print(np.mean(image[:, :, 0]))
        print(np.min(image[:, :, 0]))
        print(np.max(image[:, :, 0]))
        if max_size is not None:
            factor = float(max_size) / np.max(image.size)
            size = np.array(image.size) * factor

            print("size before: {}, as type: {}".format(size, type(image)))
            size = np.array([size.astype(int), size.astype(int), 3])
            print("image: as type: {}".format(image.shape))
            print("size after: {}, as type: {}".format(size, type(image)))
            image = transform.resize(image, size)

        if shape is not None:
            image = transform.resize(image, size)

        return np.float32(image)


    @staticmethod
    def add_one_dim(image):
        shape = (1,) + image.shape
        return np.reshape(image, shape)


    # Save an image as a jpeg-file.
    # The image is given as a numpy array with pixel-values between 0 and 255.
    @staticmethod
    def save_image(image, filename):
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert to bytes.
        image = image.astype(np.uint8)
        print(image[:10, :10, :])

        # Write the image-file in jpeg-format.
        # with open(filename, 'wb') as file:
        io.imsave(filename, image)

    @staticmethod
    def get_formatted_date():
        return datetime.today().strftime('%Y-%m-%d')
