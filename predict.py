import argparse
parse = argparse.ArgumentParser()
parse.add_argument('path_to_image')
parse.add_argument('saved_model')
parse.add_argument('--top_k', type=int, default=5)
parse.add_argument('--category_names', default='label_map.json')
args = parse.parse_args()

image = args.path_to_image
saved_model = args.saved_model
top_k = args.top_k
category_names = args.category_names

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image

image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    test_image = process_image(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)
    probs, classes = tf.math.top_k(prediction, k=top_k)
    return probs, classes

model = tf.keras.models.load_model(saved_model)
model.summary()

probs, classes = predict(image, model, top_k)

with open(category_names, 'r') as f:
    names = json.load(f)
    
print("Based on Top K Classes")
for i in range(top_k):
    print(i+1,"- Class: ",names[str(classes.numpy()[0,i]+1)])
    print("Probability: ",probs.numpy()[0,i])

print("\nThe most likely class and probability:")
print("Class: ",names[str(classes.numpy()[0,0]+1)])
print("Probability: ",probs.numpy()[0,0])