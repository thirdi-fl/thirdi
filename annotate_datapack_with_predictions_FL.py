from eval_util import getEvalArgs, iterate_multi_threaded, contentToString, isNonemptyMsg, breakUpChat,  MasterClassifier
from pathlib import Path
import json
import os
import argparse
import transformers
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timezone




parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
  "--eval_mode",
  dest='eval_mode',
  help="whether to evaluate with complete predator chats or (predator and non-predator) segments of chats. *_fast modes are recommended. They speeds up the respecitve mode by only analyzing until the first warning is raised.",
  choices=["segments", "segments_fast", "full", "full_fast"],
  required=False
)
parser.add_argument(
  "--window_size",
  dest='window_size',
  help="we look at the last `window_size` messages during classification",
  type=int,
  default=50
)

args = getEvalArgs(parser)

# get the datapack
datapackPath = "datapack-PANC-test.json"


def getTimestamp():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def getUNIXTimestamp():
	return int(datetime.now().replace(tzinfo=timezone.utc).timestamp()*1000)


with open(datapackPath, "r") as file:
  datapack = json.load(file)
chatNames = sorted(list(datapack["chats"].keys()))
# information about datapacks can be found in the chat-visualizer repo

# if in full mode, only evaluate on complete positive chats.
# because negative chats are always just segments, full mode is the same as
# segment mode for them.
if args.eval_mode.startswith("full"):
  chatNames = [name for name in chatNames
    if datapack["chats"][name]["className"] == "predator"]

torch.cuda.empty_cache()
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Load model for feature extraction of the test set
config = BertConfig.from_pretrained("config.json") 
model = BertModel.from_pretrained("bert-base-uncased-model",config=config)  
model.to(device)

tokenizer = BertTokenizer.from_pretrained("vocab.txt")

# # Load trained logistic regression
# model_path = args.model_path 
# lr_clf = pickle.load(open(model_path, 'rb'))



class LogisticRegression(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
      super(LogisticRegression, self).__init__()
      self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
      outputs = torch.sigmoid(self.linear(x))
      return outputs

#Load torch model
model_path = args.model_path 
lr_clf = LogisticRegression(768, 1)
lr_clf.load_state_dict(torch.load(model_path))
lr_clf.eval()
lr_clf.to(device)




def get_features(sentence, tokenizer):
  tokenized_text = tokenizer.encode_plus(sentence,
                                        add_special_tokens = True,
                                        max_length = 512,
                                        padding='max_length',
                                        return_attention_mask = True,
                                        return_tensors = 'pt',
                                        truncation = True
                                        )

  input_id = tokenized_text['input_ids']
  attention_mask = tokenized_text['attention_mask']
  
  with torch.no_grad():
    outputs = model(input_ids = input_id.to(device), attention_mask = attention_mask.to(device))
    cls = outputs.last_hidden_state[:,0,:].cpu().detach()
  
  return cls

def test_probas(lr_clf, embeddings, device: str):
  criterion = torch.nn.BCELoss()
  with torch.no_grad():
    output = torch.squeeze(lr_clf(embeddings.to(device)))
    preds = output.cpu().detach().numpy()
  return float(preds)


def annotateExtract(extract, classifier):

  # In fast mode, we only annotate until the master classifier with maximum
  # skepticism 10 raises a warning. Later in evaluation, if we annotate this
  # way, there will always be enough annotated messages to evaluate for all
  # skepticisms
  mc = MasterClassifier(10)

  nonempty_messages = [ct for ct in extract if isNonemptyMsg(ct)]
  for i, msg in enumerate(nonempty_messages):
    # last args.window_size messages up to message with index i
    window = nonempty_messages[max(0, i+1-args.window_size):i+1]
    text = contentToString(window)
    #pre-process the text
    feature = get_features(text, tokenizer)
    prediction = test_probas(lr_clf, feature, device)
    # prediction = classifier.predict_proba(feature)[0][1]
        
    
    # Annotate the message. This modifies the referenced message
    # that is then also modified in our datapack.
    msg["prediction"] = prediction

    # in fast mode
    mc_raised_warning = mc.add_prediction(prediction >= args.threshold)
    if mc_raised_warning and args.eval_mode.endswith("_fast"):
      return # stop annotating when warning is raised

def annotateSlice(dataset_slice, step):
  # get the classifier for the thread, initialize with existing model info
  classifier = lr_clf

  for chatName in chatNames[dataset_slice]:
    for extract in breakUpChat(datapack["chats"][chatName], args):
      annotateExtract(extract, classifier)
    step()

print("Starting work on %s chats (which might have multiple segments each)" % len(chatNames))
iterate_multi_threaded(len(chatNames), args.threads, annotateSlice)

print("all done\n")

# dump annotated datapack

suffix = "eval_mode-%s--window_size-%s" % (args.eval_mode, args.window_size)
datapack["datapackID"] += "--" + suffix
if datapack["description"] == None: datapack["description"] = ""
datapack["description"] += "Annotated with predictions " 

eval_dir = "/evaluation/results/%s/message_based_eval/" % (args.output_dir)


Path(eval_dir).mkdir(parents=True, exist_ok=True) # might not exist yet

datapack["generatedAtTime"] = getUNIXTimestamp()

# outFile = eval_dir + "annotated-datapack-%s-test-%s.json" % ( # should be --%s.json
outFile = eval_dir + "annotated-datapack-PANC-test-%s.json" % (suffix)
with open(outFile, "w") as file: json.dump(datapack, file, indent=4)



