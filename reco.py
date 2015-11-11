__author__ = 'anaviltripathi'
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import pandas as pd
import os
from scipy import misc
import glob
from PIL import Image
import numpy



#f = open('./F0000_14/C0000_14.CLS')

def cropImage(im, crop_ratio):
    imx, imy = im.shape
    return im[imx/crop_ratio: -imx/crop_ratio, imy/crop_ratio: -imy/crop_ratio]

def loadImage(image):
    #reading image using scipy
    im = misc.imread(image, flatten=1)
    per = 80
    im = cropImage(im, 3)
    im = misc.imresize(im, per)
    #misc.imshow(im)
    #print im.shape
    im = im.flatten()
    return im

def fileLister(path):
    import glob2
    files = glob2.glob(path)
    return sorted(files)


#def image_resize(image):



if __name__ == "__main__":

    dataset_list = fileLister('../character_dataset/sd_nineteen/**/D*.CLS')
    #print dataset_list

    dataset_size = 0

    #list_of_digit_files = fileLister('../character_dataset/sd_nineteen/HSF_4/**/*.bmp')

    #loadImage(list_of_digit_files[0])

    for i in range(len(dataset_list)):
        with open(dataset_list[i],'r') as f:
            dataset_size += int(f.readline()[:-1])


    print "Data size is :", dataset_size

    list_of_files = fileLister('../character_dataset/sd_nineteen/HSF_4/**/*_*_*_*_D*_*_*_*_*.bmp')

    #print list_of_files[:1000]
    labels = []
    for file_name in list_of_files:
        labels.append(file_name.split('_')[-2])


    #print labels
    #
    #

    image_name = list_of_files[0]

    initializer_image = loadImage(image_name)
    #print(len(initializer_image))

    net = buildNetwork(len(initializer_image), 15, 10)
    print("network built")


    ds = SupervisedDataSet(len(initializer_image) ,10)
    for i in range(dataset_size):
        ds.addSample(loadImage(list_of_files[i]), (ord(labels[i]),))
    print("data added")

    trainer = BackpropTrainer(net, ds)
    print("trainer initialized")

    error = 10
    iteration = 0

    #exit(0)
    while iteration < 10:
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)

    print "\nResult: ", net.activate(loadImage(list_of_files[0]))



