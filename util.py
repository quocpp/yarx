import matplotlib.pyplot as plt
import os
import numpy as np
def plot_images(range_in,dataframe,trainset):
    print("plot images")
    img_dir = "CXR8/images/images"
    plt.figure(figsize = (15,15))
    for i in range(range_in):
        plt.subplot(4, 4, i+1)
        plt.imshow(plt.imread(os.path.join(img_dir, trainset["Image"][i])), cmap = "gray")
        plt.title(dataframe[dataframe["Image Index"] == trainset["Image"][i]].values[0][1])
    plt.tight_layout()
    plt.show()

def analyse_sample_mage(dataframe,trainset):
    img_dir = "CXR8/images/images"
    num = np.random.randint(trainset.shape[0])
    sample = plt.imread(os.path.join(img_dir, trainset.iloc[[num]]["Image"].values[0]))
    plt.figure(figsize=(15, 15))
    plt.title(dataframe[dataframe["Image Index"] == trainset.iloc[[num]]["Image"].values[0]].values[0][1])
    plt.imshow(sample, cmap='gray')
    plt.colorbar()
    print(trainset.iloc[[num]])
    plt.show()