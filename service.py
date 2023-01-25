import os
# import logging
import tempfile
# import tensorflow as tf
# from keras.models import load_model
# from keras.models import Model
# from tensorflow import keras
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import subprocess
# own
from utils.grayscale import clear_captcha
# from utils.cnn import encode_single_sample, decode_batch_predictions

app = Flask(__name__)
app.test_client()

FILE_UPLOAD_HTML = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

ALLOWED_EXTENSIONS_IMAGE = {'jpg', 'jpeg', 'png', 'webp', 'tiff', 'bmp'}
ct_json = {'Content-Type': 'application/json'}

# ---- start subprocess with tensorflow
# -- unset PYTHONPATH for Tensorflow - it uses own numpy version
my_env = os.environ.copy()
my_env["PYTHONPATH"] = ""
pipe = subprocess.Popen(["python3", "./service_process.py"], text=True, universal_newlines=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env)
if pipe.returncode is not None:
    raise Exception("no subprocess service tensorflow")
while pipe.stdout.readline() != 'ready\n':  # wait for ready response
    pass


def allowed_file(filename):
    ext = os.path.splitext(filename)
    # latin and unicode
    return (ext[1][1:].lower() in ALLOWED_EXTENSIONS_IMAGE) or (ext[1] == '' and ext[0].lower() in ALLOWED_EXTENSIONS_IMAGE)

# -- TENSORFLOW --

# MODEL_PATH = './m_ctc'
#
# try:
#     gpus = tf.config.experimental.list_physical_devices('CPU')
#     tf.config.set_visible_devices(gpus[0], 'CPU') if gpus else print('no gpu!!!!!!!!!!!!!!!!!!!!:', gpus)
# except RuntimeError as e:
#     print(e)
#
# # disable logger
# logging.getLogger('tensorflow').disabled = True
#
# # -- load model
# model: Model = load_model(MODEL_PATH, compile=False)
# prediction_model = keras.models.Model(
#         model.get_layer(name="image").input, model.get_layer(name="dense2").output
#     )

# max_length = 6  # 4-6


@app.route('/captcha_image', methods=['GET', 'POST'])
def captcha_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file or not allowed_file(file.filename):
            return {'error': 'Bad Request, file'}, 400, ct_json

        filename = secure_filename(file.filename)
        with tempfile.TemporaryDirectory() as tmpdir:
            # -- save input file
            fp = os.path.join(tmpdir, filename)
            file.save(fp)
            img: np.ndarray = cv.imread(fp)
            gray = clear_captcha(img)
            cv.imwrite(fp, gray)
            # -- send file to subprocess with Tensorflow
            pipe.stdin.write(fp + "\n")
            pipe.stdin.flush()

            r = pipe.stdout.readline()
            if not r.startswith('answer'):
                return {'error': 'Error in subprocess Tensorflow'}, 500, ct_json
            label = r[:-1][7:]

            # encsample = encode_single_sample(fp, '2222')
            # img2 = tf.expand_dims(encsample["image"], axis=0)
            #
            # pred = prediction_model.predict(img2, use_multiprocessing=True, verbose=False)
            # pred_label = decode_batch_predictions(pred, max_length)
            # pred_label = pred_label[0].split('[')[0]
            return label

    elif request.method == 'GET':
        return FILE_UPLOAD_HTML, 200, {'Content-Type': 'text/html'}


if __name__ == '__main__':
    app.run(debug=False)
