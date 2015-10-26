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

def loadImage(image):
    #reading image using scipy
    im = misc.imread(image, flatten=1)
    im = im.flatten()
    #image_size = len(im)
    #new_image = numpy.zeros(128*128, dtype=float)
    #print(len(new_image))
    #print(image_size)
    #for l in im:
    #    print("l ",l.flatten()[:])
    #    new_image

    #new_image.flatten()
    #print(new_image)
    #pix = numpy.array(im)
    #misc.imshow(image)

    #normal reading without using any special libraray
    #with open(image, 'rb') as f:
    #    im = bytearray(f.read())


    #writing into a new bmp file
    #f = open('newfile.bmp', 'wb')
    #f.write(image)


    #reading using PIL
    #im = Image.open(image)

    #print(pix)

    #print(im)
    #print image #prints something incomprehensible

    return im
    #image.rotate(180).show()

    #writing into a new bmp file
    #f = open('newfile.bmp', 'wb')
    #f.write(image)

    #dataset size calculation

def fileLister(path):
    return glob.glob(path)





if __name__ == "__main__":

    dataset_list = fileLister('./F0000_14/*.CLS')
    #print dataset_list
    dataset_size = 0

    for i in range(len(dataset_list)):
        with open(dataset_list[i],'r') as f:
            dataset_size += int(f.readline()[:-1])


    #print "Data size is :", dataset_size
    list_of_files = fileLister('./F0000_14/*.bmp')
    labels = []
    for file_name in list_of_files:
        labels.append(file_name.split('_')[-2])

    #print(labels)
    image_name = list_of_files[0]

    initializer_image = loadImage(image_name)
    #print(len(initializer_image))

    net = buildNetwork(len(initializer_image), 100, 62)

    ds = SupervisedDataSet(len(initializer_image) ,62)
    for i in range(dataset_size):
        ds.addSample(loadImage(list_of_files[i]), (ord(labels[i]),))

    trainer = BackpropTrainer(net, ds)

    error = 10
    iteration = 0
    while error > 0.001:
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)

    print "\nResult: ", net.activate(loadImage(list_of_files[0]))