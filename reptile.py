import os
import requests
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from keras import backend as K
from keras.metrics import Precision, Recall
from tensorflow import keras


base_dir = r'C:\Users\Cauchy\Desktop\images'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

animals = ['bird', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'panda', 'zebra', 'tiger', 'tortoise']

base_url = 'https://api.pexels.com/v1/search'
headers = {'Authorization': 'cCEBAEAMvX4QQEBF9PqjP6a7i2aoFlEOxIWmdWmSJ9M0gXKjDmQnL5hw'}

for animal in animals:
    save_dir = os.path.join(base_dir, animal)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url = f'{base_url}?query={animal}&per_page=100'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for i, photo in enumerate(data['photos']):
            photo_url = photo['src']['large']
            try:
                response = requests.get(photo_url)
                filepath = os.path.join(save_dir, f"{animal}_{i+1}.jpg")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded image {i+1} of {animal}")
            except Exception as e:
                print(f"Failed to download image {i+1} of {animal}. Error: {e}")
    else:
        print(f"Failed to download images of {animal}. Status code: {response.status_code}")