import pandas as pd
import numpy as np
import json
import pickle
import re
import string

#Load the training set
df_train = pd.read_csv("/eSPD-datasets/PANC/csv/PANC-train.csv", encoding="latin-1")
df_train["label"] = np.where(df_train["label"]=="non-predator",0,1)



#Load the training set with additional chunks of data TVT
train_tvt = pd.read_csv("train_tvt.csv")

#Add a "name" column
name_list=[]
for index, row in train_tvt.iterrows():
  if (row['type'].find('chunk') == -1) & (row['label']==1) :
    train_tvt.at[index,'name'] = row['chatName'].rsplit('-',1)[0]  
  else:
    train_tvt.at[index,'name'] = row['chatName']

#Count the number of segments associated to each userID
count_df = train_tvt.groupby('name')['label'].count().reset_index().rename(columns={'label':'count'})
label_count_df = pd.merge(count_df, train_tvt[['name','label']], how='left', left_on='name', right_on='name')
label_count_df = label_count_df.drop_duplicates()
#Shuffle the data
label_count_df.sample(frac=1)

#Create a training set without augmented data
train_tvt_segment = train_tvt[train_tvt['type']=='normal']

#Randomly select a certain number of userID while making sure that the sum is inferior to totalRows.
#This function allows us to randomly sample data for all the different data splits (train, validation, warm-up) while ensuring that the data of the same userID is not split
#And that the % of data in each split is respected
def random_selection(df, totalRows):
  user_list = []
  rows = 0
  for i in df.sample(frac=1).iterrows():
      if (rows + i[1]['count']) <= totalRows:
          rows += i[1]['count']
          user_list.append(i[1]['name'])
      if rows == totalRows:
          break
  return user_list


#Create warm-up data : We want it to represent 10% of the complete dataset and to be balanced (5% negative labels and 5% positive labels)
num_pos_labels_warmup = 20343 * 10/100/2   #20343 is the total number of rows in the dataset
num_neg_labels_warmup = 20343 * 10/100/2 
pred_warmup_list = random_selection(label_count_df[label_count_df['label']==1], num_pos_labels_warmup)
nonpred_warmup_list = label_count_df[label_count_df['label']==0].sample(n=int(num_neg_labels_warmup))['name'].to_list()
warmup_list = pred_warmup_list + nonpred_warmup_list


#Create validation set: 10% of what is left of the data with a 9%-91% data distribution as in the test set
label_count_df_warmup = label_count_df[~label_count_df['name'].isin(warmup_list)]
num_pos_labels_val = len(train_tvt_segment[~train_tvt_segment['name'].isin(warmup_list)])*0.10*0.09
num_neg_labels_val = len(train_tvt_segment[~train_tvt_segment['name'].isin(warmup_list)])*0.10*0.91
pred_val_list = random_selection(label_count_df_warmup[label_count_df_warmup['label']==1], num_pos_labels_val)
nonpred_val_list = label_count_df_warmup[label_count_df_warmup['label']==0].sample(n=int(num_neg_labels_val))['name'].to_list()
val_list = pred_val_list + nonpred_val_list

#Create training set =) everything else
train_list = label_count_df_warmup[~label_count_df_warmup['name'].isin(val_list)]['name'].to_list()

#Create a list of non-predatory users
nonpred_list = df_train[df_train["label"]==0]["chatName"].tolist()



#Download the files
with open('warmup_list.pkl', 'wb') as i:
  pickle.dump(warmup_list, i)
with open('train_list.pkl', 'wb') as l:
  pickle.dump(train_list, l)
with open('val_list.pkl', 'wb') as e:
  pickle.dump(val_list, e)
with open('nonpred_list.pkl', 'wb') as e:
  pickle.dump(nonpred_list, e)

