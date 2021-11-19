from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
# import cv2

app = Flask(__name__)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.load_weights('./checkpoints/checkpoint')
print("+"*21, "Model is Ready!!", "+"*21)

labels = pd.read_csv("labels.txt", sep='\n').values
# print(labels)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img']
    img.save("./data/predict_img.jpg")
    # image = cv2.imread("img.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(img)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.reshape(image, (1, 28, 28, 3))
    
    pred = np.argmax(model.predict(image))
    pred = labels[pred]
    
    return render_template('prediction.html', data=pred)

                         
if __name__ == "__main__":
    app.run(debug=True)