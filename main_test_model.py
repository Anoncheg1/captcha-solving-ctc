import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
from utils.grayscale import clear_captcha
import tempfile
from keras.models import load_model
from keras.models import Model
import logging
from tqdm import tqdm
# own
from utils.cnn import image_tensorflow_prepare, ALPHABET
from cnncaptcha.main_cnn import num_to_char

validate = './jpg2'
# validate = './phptest/gen_train'
model_path = './m'

# -- set device manually
try:
    gpus = tf.config.experimental.list_physical_devices('CPU')
    tf.config.set_visible_devices(gpus[0], 'CPU') if gpus else print('no gpu!!!!!!!!!!!!!!!!!!!!:', gpus)
except RuntimeError as e:
    print(e)

# disable logger
logging.getLogger('tensorflow').disabled = True


# -- load model
model: Model = load_model(model_path, compile=False)
prediction_model = keras.models.Model(
        model.get_layer(name="image").input, (
        model.get_layer(name="reshape2").output,  # characters
        model.get_layer(name="dense12").output)  # count of characters
    )
prediction_model.summary()


max_length = 6  # 4-6

c = 0

for i, filename in tqdm(enumerate(os.listdir(validate))):
    # print(i, filename)
    # nsolv = filename[0:5]
    label = filename.split('.')[0]
    file = os.path.join(validate, filename)
    # inv = [4, 8, 67, 72, 80, 95]
    # -- image to grayscale and save to file
    img: np.ndarray = cv.imread(file)
    gray = clear_captcha(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        # -- prepare image for model
        img_path = os.path.join(tmpdir, filename)
        cv.imwrite(img_path, gray)
        img = image_tensorflow_prepare(img_path)
        pred = prediction_model.predict(np.array([img]), use_multiprocessing=True, verbose=False)
        chars, count = pred
        count = np.round(count).tolist()[0]
        # print(count, [1., 0.])
        chars = np.argmax(chars, axis=2)[0]
        # print(chars)
        chars = ''.join([ALPHABET[x] for x in chars])
        # print(chars)

        if count == [1., 0.]:
            length = 5
        elif count == [0., 0.]:
            length = 4
        else:
            length = 6
        # print(chars.shape)
        chars = chars[:length]
        if chars != label:
            print(label, chars, False)

        c += chars == label

        # c += num_to_char(pred) == filename[:-4]
        # print(filename)
        # m.predict(img)

print(c, len(os.listdir(validate)), c / len(os.listdir(validate)))
