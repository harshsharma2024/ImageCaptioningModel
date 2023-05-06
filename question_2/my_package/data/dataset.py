# Imports
import jsonlines
import json
from PIL import Image
import os
import numpy as np


class Dataset(object):
    '''
        A class for the dataset that will return data items as per the given index
    '''

    def __init__(self, annotation_file, transforms=None):
        '''
            Arguments:
            annotation_file: path to the annotation file
            transforms: list of transforms (class instances)
                        For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
        '''
        self.transforms = transforms
        with open(annotation_file, 'r+') as jsonFile:
           jsonList = list(jsonFile)
        self.jsonList = []
        for jsonStr in jsonList:
            self.jsonList.append(json.loads(jsonStr))
     

    def __len__(self):
        '''
            return the number of data points in the dataset
        '''
        return len(self.jsonList)

    
    def __getann__(self, idx):
        '''
            return the data items for the index idx as an object
        '''
        dataIdx=self.jsonList[idx]
        return dataIdx
        

    def __transformitem__(self, path):
        '''
            return transformed PIL Image object for the image in the given path
        '''
        img=Image.open(path)
        if self.transforms != None:
            for instanceTransform in self.transforms:
                img = instanceTransform(img)
        return img

