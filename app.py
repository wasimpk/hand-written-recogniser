import numpy as np
from keras.models import load_model
import cv2
import io
import base64
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO
import json
import matplotlib.pyplot as plt
app = Flask(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@app.route('/check_char', methods=['POST'])
def check_char():
    model = load_model('model_hand_updated.h5')
    req = request.get_json()
    file = req['img']
    bytes_img = Image.open(BytesIO(base64.b64decode(file)))
    np_img = np.array(bytes_img)
    img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    img = img_final // 255
    # ====================================
    # ====================================
    fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    axes = axes.flatten()
    img_pred = model.predict(img_final)
    result = {"Predictions": img_pred,
              "img": img_final.shape,
              "size": img_final.size,
              "char_index": np.argmax(img_pred)
              }
    return json.dumps(result, cls=NpEncoder)


if __name__ == '__main__':

    app.run(debug=True)
