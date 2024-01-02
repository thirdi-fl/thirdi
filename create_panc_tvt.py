import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import numpy as np
import csv
import sys
import os
import json
import random
import re


#Load the training set
df_train = pd.read_csv("/eSPD-datasets/PANC/csv/PANC-train.csv", encoding="latin-1")

#Create 2 columns: "name" and "num" to be able to identify and group all the predatory segments from the same predator 
df_train[['name','num']] = df_train['chatName'].str.rsplit('-',n=1,expand=True)

#Transform label from strings ('predator', 'non-predator') to binary variable 
df_train['label'] = np.where(df_train['label']=='predator',1,0)

#Group all the segments assigned to the same user into the same cell and only keep one instance per user
df_predators = df_train[df_train['label']==1]
df_predators['text'] = df_predators.groupby(['name'])['segment'].transform(lambda x: ' '.join(x))
data_to_split = df_predators[['name','text']]
data_to_split_wdupes = data_to_split.drop_duplicates()

#Create a function to split the data by chunks
def get_chunk(col_list, size):
  segment_list = []
  for segment in col_list:
      ls = re.findall("[\w]+[']?[\w]*", segment) #find all the words
      idxl = round(size*len(ls))
      arr_str = ls[0:idxl]
      segment_list.append(' '.join(arr_str))
  return segment_list

#Create the different chunks needed for the data augmentation
data_to_split_wdupes['chunk1'] = get_chunk(data_to_split_wdupes.text.to_list(), 0.10) #10% of the full conversation
data_to_split_wdupes['chunk2'] = get_chunk(data_to_split_wdupes.text.to_list(), 0.20) #20% of the full conversation
data_to_split_wdupes['chunk3'] = get_chunk(data_to_split_wdupes.text.to_list(), 0.30) #30% of the full conversation
data_to_split_wdupes['chunk4'] = get_chunk(data_to_split_wdupes.text.to_list(), 0.40) #40% of the full conversation

#Put all the newly created columns into the data segment. This create a new column: variable with either text or chunk1, chunk2, chunk3, chunk4 as a value
data_melt = pd.melt(data_to_split_wdupes, id_vars=['name'], value_name='segment')

#Isolate the newly created chunks and assigned the label 1 to them abd add a column "type" to be able to find back the chunks once they are merged
data_tvt = data_melt[data_melt['variable'].str.contains('chunk')][['name','segment']]
data_tvt['label'] = 1
data_tvt["type"] = "tvt"
#Add a column type to the initial training set before merging the new chunks of data
df_train["type"] = "normal"

#Concatenate the training set with the newly added chunks
train = df_train[['label','chatName','segment','type']]
data_tvt['chatName'] = data_tvt['name']
data_tvt_tomerge = data_tvt[['label','chatName','segment','type']]

data_train_tvt = pd.concat([train, data_tvt_tomerge])

#Export the new training set with TVT
data_train_tvt.to_csv("train_tvt.csv")