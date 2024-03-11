import pandas as pd
import os
from matplotlib import rcParams
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import tensorflow.keras.backend as K
from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.python.framework.ops import disable_eager_execution
import random
import model as model1
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
#from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def process_data():
    dataframe = pd.read_csv("CXR8/Data_Entry_2017_v2020.csv")
    columns = ["Image"]
    for i in dataframe["Finding Labels"].values:
        for j in i.split("|"):
            if j not in columns:
                columns.append(j)
    labels = columns.copy()
    labels.remove("Image")
    labels.remove("No Finding")
    columns.remove("No Finding")
    return labels, columns, dataframe
def get_mean_std_per_batch(image_path, df, H=400, W=400):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def load_image(img, image_dir, df, preprocess=True, H=400, W=400):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
        print(f"mean: {mean}, std: {std}")
    return x
def grad_cam(input_model, image, cls, layer_name, H=400, W=400):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output

    #grads = K.gradients(y_c, conv_output)[0]
    grads = tf.compat.v1.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def sort_result(res_in):
    res = sorted(res_in.items(),key=lambda x:x[1],reverse=True)
    return res
def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    res_dict = {}
    preprocessed_input = load_image(img, image_dir, df)
    print(type(preprocessed_input))
    predictions = model.predict(preprocessed_input)
    rows = 4
    cols = 2
    print("Loading original image")
    #plt.figure(figsize=(15, 10))
    fig = plt.figure(figsize=(15, 10))
    #plt.subplot(151)
    fig.add_subplot(rows, cols, 1)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')
    j = 2
    for i in range(len(labels)):
        res_dict[i] = predictions[0][i]
    res = sort_result(res_dict)
    converted_dict = dict(res)
    print(converted_dict)
    count = 0
    for label_idx in converted_dict:
        print(label_idx)
        gradcam = grad_cam(model, preprocessed_input, int(label_idx), layer_name)
        fig.add_subplot(rows, cols, j)
        count = count + 1
        plt.title(f"{labels[int(label_idx)]}: p={predictions[0][int(label_idx)]:.3f}")
        plt.axis('off')
        plt.imshow(load_image(img, image_dir, df, preprocess=False),
                            cmap='gray')
        plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][int(label_idx)]))
        j += 1
        if count >= 3:
            break
    # for i in range(len(labels)):
    #     print(predictions[0][i])
    #     if predictions[0][i] > 0.7:
    #         gradcam = grad_cam(model, preprocessed_input, i, layer_name)
    #         fig.add_subplot(rows, cols, j)
    #         plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
    #         plt.axis('off')
    #         plt.imshow(load_image(img, image_dir, df,preprocess=False),
    #                    cmap='gray')
    #         plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
    #         j += 1
    #         if j >= 8:
    #             break
    plt.show()

def prepare_data_set(columns,dataframe,from_idx, to_idx):
    print("start prepare dataset")
    trainset = pd.DataFrame(columns = columns)
    for i in range(from_idx,to_idx):
        col = [0]*len(columns)
        col[0] = dataframe["Image Index"][i]
        count = 1
        for j in columns[1:]:
            if(j in dataframe["Finding Labels"][i]):
                col[count] = 1
            count+=1
        trainset.loc[len(trainset)] = col
    print("finished prepare dataset")
    return trainset

def run_wrapper():
    labels, columns, dataframe = process_data()
    #trainset = prepare_data_set(columns, dataframe, 0, 90000)
    trainset = prepare_data_set(columns, dataframe, 0, 900)
    model = model1.create_model(labels)
    model.load_weights("cxr_q1.h5")
    #compute_gradcam(model, '00000001_001.png', "CXR8/images/images/", trainset, labels, labels)
    #compute_gradcam(model, '00000013_024.png', "CXR8/images/images/", trainset, labels, labels)
    compute_gradcam(model, sys.argv[1], "CXR8/images/images/", trainset, labels, labels)

run_wrapper()