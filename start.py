
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from keras import backend as K
from tensorflow import keras

# 自定义 precision 指标函数
def precision(y_true, y_pred):
    # 计算精确率
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    # 计算召回率
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
# 加载模型时注册 precision 指标函数
with keras.utils.custom_object_scope({'precision': precision},{'recall': recall}):
    model = tf.keras.models.load_model('F:\\lunwen\\moxing1.h5')



# 定义动物类型
class_labels = ['dog','horse','elephant','Butterfly','Cock','cat','cow','sheep','spider','Squirrel']                                                                                                 

# 加载模型

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224,224), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        predict_image(file_path)


def predict_image(file_path):

    img = Image.open(file_path)
    img = img.resize((224,224), Image.ANTIALIAS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    probs = model.predict(img_array)[0]

 
    for i in range(len(class_labels)):
        result_label.config(text=result_label.cget('text') + "\n" + class_labels[i] + ": " + str(round(probs[i]*100, 2)) + "%")





window = tk.Tk()
window.title("Animal Classification")


image_label = tk.Label(window)
image_label.pack()


select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack()


result_label = tk.Label(window, text="Prediction Results:")
result_label.pack()

def predict_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224,224), Image.ANTIALIAS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    probs = model.predict(img_array)[0]

    # 先清空之前的文本内容
    result_label.config(text="Prediction Results:")

    for i in range(len(class_labels)):
        result_label.config(text=result_label.cget('text') + "\n" + class_labels[i] + ": " + str(round(probs[i]*100, 2)) + "%")


window.mainloop()