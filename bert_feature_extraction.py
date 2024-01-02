from transformers import BertTokenizer, BertConfig, BertModel
import torch
import pandas as pd
import numpy as np
import pickle



#Load the training set with additional chunks of data TVT
train_tvt = pd.read_csv("train_tvt.csv")

device = torch.device('cuda')
torch.cuda.empty_cache()


#Load the tokenizer and the Bert Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=False)
model = BertModel.from_pretrained("bert-base-uncased", config=config)
gpu_model = model.to(device)


#Transform the data into input_ids and attention_mask
def encode(textCol, tokenizer):
  input_ids = []
  attention_mask = []
  for text in textCol:
    tokenized_text = tokenizer.encode_plus(text,
                                          add_special_tokens = True,
                                          max_length = 512,
                                          pad_to_max_length = True,
                                          return_attention_mask = True,
                                          return_tensors = 'pt',
                                          truncation=True
                                           )
    input_ids.append(tokenized_text['input_ids'])
    attention_mask.append(tokenized_text['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_mask = torch.cat(attention_mask, dim=0)
    
  return input_ids, attention_mask


#Create tensor dataloader
def get_batches(textCol, labelCol, indexCol, tokenizer, batch_size=64):
    x = list(textCol.values)
    
    y_indices = list(labelCol)
    index_indices = list(indexCol)
    y = torch.tensor(y_indices, dtype=torch.long)
    index = torch.tensor(index_indices, dtype=torch.long)
    input_ids, attention_mask = encode(x, tokenizer)
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y, index)
    tensor_randomsampler = torch.utils.data.RandomSampler(tensor_dataset)
    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, sampler=tensor_randomsampler, batch_size=batch_size)
    return tensor_dataloader


train_dataloader = get_batches(train_tvt['segment'],train_tvt['label'], train_tvt['Unnamed: 0'], tokenizer, batch_size=8)


#Only keep the last hidden layer
with torch.no_grad():
  batches = []
  for i, batch_tuple in enumerate(train_dataloader):
    input_ids, attention_mask, labels, index = batch_tuple
    outputs = gpu_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)) #forward does not take labels as an input
    cls = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()

    batches.append((index, labels, cls))

#Flatten the output
index_list = []
labels_list = []
embeddings_list = []
for index, labels, embeddings in batches:
  index_list.append(index.numpy())
  labels_list.append(labels.numpy())
  embeddings_list.append(embeddings)


#flatten the labels list
all_labels = []
for labels in labels_list:
  for label in labels:
    all_labels.append(label)

#flatten the index list
all_index = []
for indexes in index_list:
  for index in indexes:
    all_index.append(index)

#flatten the embeddings list
all_embeddings = []
for embeddings in embeddings_list:
  for embedding in embeddings:
    all_embeddings.append(embedding)


#Find the chatnames using the index
df_index = pd.DataFrame(all_index, columns=["index"])
df_merge = pd.merge(df_index, train_tvt[['Unnamed: 0','chatName']], how="left", left_on="index", right_on="Unnamed: 0")
all_names = df_merge.chatName.to_list()

with open('all_embeddings.pkl', 'wb') as i:
  pickle.dump(all_embeddings, i)
with open('all_labels.pkl', 'wb') as l:
  pickle.dump(all_labels, l)
with open('all_names.pkl', 'wb') as e:
  pickle.dump(all_names, e)