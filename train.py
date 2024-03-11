import pandas as pd
import os
from matplotlib import rcParams
#import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.python.framework.ops import disable_eager_execution
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import util as util
import model as model1
from keras.preprocessing.image import ImageDataGenerator

IMAGE_DIR = "CXR8/images/images/"
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

def prepare_data_set(columns,dataframe,from_idx, to_idx):
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
    return trainset

def prepare_val_set(columns,dataframe):
    print("prepair test set")
    valset = pd.DataFrame(columns = columns)
    for i in range(90000, 100000):
        col = [0]*len(columns)
        col[0] = dataframe["Image Index"][i]
        count = 1
        for j in columns[1:]:
            if(j in dataframe["Finding Labels"][i]):
                col[count] = 1
            count+=1
        valset.loc[len(valset)] = col
    return valset

def prepare_test_set1(columns,dataframe):
    print("prepair test set1")
    testset = pd.DataFrame(columns = columns)
    for i in range(50000, 50500):
        col = [0]*len(columns)
        col[0] = dataframe["Image Index"][i]
        count = 1
        for j in columns[1:]:
            if(j in dataframe["Finding Labels"][i]):
                col[count] = 1
            count+=1
        testset.loc[len(testset)] = col
    return testset

def prepare_test_set(columns,dataframe):
    testset = pd.DataFrame(columns = columns)
    for i in range(100000, 112000):
        col = [0]*len(columns)
        col[0] = dataframe["Image Index"][i]
        count = 1
        for j in columns[1:]:
            if(j in dataframe["Finding Labels"][i]):
                col[count] = 1
            count+=1
        testset.loc[len(testset)] = col
    return testset

def test():
    labels, columns, dataframe = process_data()
    #trainset = prepare_data_set(columns, dataframe,0,90000)
    trainset = prepare_data_set(columns, dataframe,0,900)
    #valset = prepare_data_set(columns,dataframe,90000,100000)
    valset = prepare_data_set(columns,dataframe,900,1500)
    testset = prepare_test_set(columns,dataframe)
    testset1 = prepare_test_set1(columns,dataframe)
    #util.plot_images(16,dataframe,trainset)
    util.analyse_sample_mage(dataframe,trainset)

def train():
    labels, columns, dataframe = process_data()
    #trainset = prepare_data_set(columns, dataframe, 0, 900)
    trainset = prepare_data_set(columns, dataframe, 0, 90000)
    #valset = prepare_data_set(columns, dataframe, 900, 1500)
    valset = prepare_data_set(columns, dataframe, 90000, 100000)
    traingen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    traingenerator = traingen.flow_from_dataframe(
        dataframe=trainset,
        directory=IMAGE_DIR,
        x_col="Image",
        y_col=labels,
        class_mode="raw",
        batch_size=1,
        shuffle=True,
        target_size=(512, 512)
    )
    imagegen = ImageDataGenerator().flow_from_dataframe(dataframe=trainset, directory=IMAGE_DIR, x_col="Image",
                                                        y_col=labels, class_mode="raw", batch_size=1, shuffle=True,
                                                        target_size=(512, 512))
    train_sample = imagegen.next()[0]
    imagegen1 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    imagegen1.fit(train_sample)
    valgenerator = imagegen1.flow_from_dataframe(dataframe = valset, directory = IMAGE_DIR, x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=True, target_size=(512,512))

    model = model1.create_model(labels)
    fitter = model.fit(traingenerator, validation_data=valgenerator, steps_per_epoch=1000, epochs=50)
    model.save_weights("cxr_quoc.h5")

train()