from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from keras import backend as K
from werkzeug.utils import secure_filename
matplotlib.use('Agg')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPEG', 'PNG', 'JPG'}
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = 'Munger1317'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_prep(image):
  img_size = (256, 256)
  model_img = np.zeros((1,) + img_size + (3,), dtype='float32')
  img = load_img(image, color_mode='rgb', target_size=img_size)
  img = img.crop((30, 50, 230, 220))
  img = img.resize((256, 256))
  model_img[0] = img_to_array(img)
  return model_img


# Set up functions to calculate Focal Tversky Loss
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_coef


def dice_coef_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    return tversky


def tversky_loss(y_true, y_pred):
    tversky_loss = 1 - tversky(y_true, y_pred)
    return 


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    focal_tv_loss = K.pow((1 - tv), gamma)
    return focal_tv_loss

teeth_model = keras.models.load_model('teeth_segmentation.h5', custom_objects={'focal_tversky_loss': focal_tversky_loss})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

new_filename = None

@app.route('/')
def index():
  files = os.listdir(app.config['UPLOAD_FOLDER'])
  return render_template('index.html', files=files)

# Upload file and save to disk
@app.route('/', methods=['POST'])
def upload_file():
    global new_filename
    if 'file' not in request.files:
      flash('No file found')
      return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      new_filename = 'mask' + filename
      path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(path)
      transformed_img = load_prep(path)
      prediction_img = teeth_model.predict(transformed_img)
      plt.imshow(load_img(path, target_size=(256,256)).crop((30, 50, 230, 220)).resize((256,256)), cmap='gray', interpolation=None)
      plt.imshow(prediction_img[0], cmap='jet', alpha=0.5, interpolation=None)
      plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
      return redirect(url_for('upload_file', filename=new_filename))
    return redirect('/display')

@app.route('/uploads/<filename>')
def upload(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], new_filename)

# Display radiograph with predicted mask on new page
@app.route('/display')
def display():
  global new_filename
  return render_template('mask.html', user_image=new_filename)

@app.after_request
def add_header(r):
 
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(debug=True)
