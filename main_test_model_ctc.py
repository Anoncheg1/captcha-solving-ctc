import time
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
# from utils.cnn import image_tensorflow_prepare
# from cnncaptcha.main_ctc_keras_original import encode_single_sample, decode_batch_predictions
from utils.cnn import encode_single_sample, decode_batch_predictions
# from cnncaptcha.sequence import ALPHABET

validate = './jpg2'
# validate = './phptest/gen_train'
model_path = './m_ctc'

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
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
prediction_model.summary()


max_length = 6  # 4-6

c = 0

for i, filename in tqdm(enumerate(os.listdir(validate))):
    print(i, filename)
    # exit()
    label = filename.split('.')[0]

    start_time = time.time()
    file = os.path.join(validate, filename)
    # inv = [4, 8, 67, 72, 80, 95]
    # -- image to grayscale and save to file
    img: np.ndarray = cv.imread(file)
    gray = clear_captcha(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        # -- prepare image for model
        img_path = os.path.join(tmpdir, filename)
        cv.imwrite(img_path, gray)
        encsample = encode_single_sample(img_path, '2222')
        img2 = tf.expand_dims(encsample["image"], axis=0)

        # img = image_tensorflow_prepare(img_path)
        pred = prediction_model.predict(img2, use_multiprocessing=True, verbose=False)
        pred_label = decode_batch_predictions(pred, max_length)
        pred_label = pred_label[0].split('[')[0]
        print(pred_label, label, pred_label == label)
        print(f"{(time.time() - start_time):.2} senconds")
        c += pred_label == label
        # format()
        # print(filename)
        # m.predict(img)

print(c, len(os.listdir(validate)), c / len(os.listdir(validate)))
