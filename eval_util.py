import threading
from tqdm import tqdm
import concurrent.futures

import argparse
from argparse import Namespace
import json
import numpy as np

def getEvalArgs(parser=None):
	if parser is None:
		parser = argparse.ArgumentParser(description='Evaluate a model')
	# parser.add_argument(
	# 	"--run",
	# 	dest='run_id',
	# 	help="the run ID you want to evaluate",
	# 	required=True
	# )
	# parser.add_argument(
	# 	"--model_version",
	# 	dest='model_version',
	# 	choices=["quantized", "non_quantized"],
	# 	help="whether to evluate with the quantized or non quantized version",
	# 	required=True
	# )
	parser.add_argument(
		"--threads",
		dest='threads',
		type=int,
		help="the number of threads to use for evaluation",
		default=5,
		required=False
	)
	parser.add_argument(
		"--split",
		dest='split',
		choices=["train", "test"],
		help="whether to use the train or the test split",
		default="test"
	)
	parser.add_argument(
		"--model_path",
		dest='model_path',
		help="path to the model you are evaluating"
	)

	parser.add_argument(
		"--output_dir",
		dest='output_dir',
		help="version of the model you are evaluating"
	)

	parser.add_argument(
		"--threshold",
		dest='threshold',
		help="classification threshold",
		type=float,
		default=0.50
	)

	args = parser.parse_args()

	# run_settings = json.load(open("/lustre06/project/6006766/khaoulac/bert-hyperparameteroptimization/%s/run_settings.json" % args.run_id))
	# run_settings = {**vars(args), **run_settings}

	# # TODO refactor
	# if run_settings["dataset_indicator"] == "VTPAN_no_sep":
	# 	run_settings["dataset_indicator"] = "VTPAN"

	# print("\n---            Run settings            ---")
	# for key, val in run_settings.items(): print("%20s: %s" % (key, val))

	# if run_settings["project"] == "flair":
	# 	assert run_settings["model_version"] == "non_quantized", \
	# 		"flair only has non_quantized models"

	# return Namespace(**run_settings)
	return args


class Score:
	# nice utility for calculating classification scores
	# all precision/recall/F are given for the positive class

	def __init__(self, tp=0, fn=0, tn=0, fp=0):
		self.tp = tp
		self.fn = fn
		self.tn = tn
		self.fp = fp

	# very cool
	def add_prediction(self, predicted_positive, is_positive):
		if   is_positive and predicted_positive:         self.tp += 1
		elif is_positive and not predicted_positive:     self.fn += 1
		elif not is_positive and not predicted_positive: self.tn += 1
		elif not is_positive and predicted_positive:     self.fp += 1

	@property
	def number_of_samples(self):
		return self.tp + self.fn + self.tn + self.fp

	@property
	def f1(self):
		return self.f_beta(1)

	def f_beta(self, beta=1):
		if self.recall+self.precision == 0: return 0
		return (1 + beta**2.0) * (self.precision * self.recall) / \
			   (beta**2.0 * self.precision + self.recall)

	@property
	def accuracy(self):
		if self.tp + self.tn + self.fp + self.fn == 0: return 0
		return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

	@property
	def precision(self):
		if self.tp + self.fp == 0: return 0
		return self.tp / (self.tp + self.fp)

	@property
	def recall(self):
		if self.tp + self.fn == 0: return 0
		return self.tp / (self.tp + self.fn)

	@property
	def false_positive_rate(self):
		if self.fp + self.tn == 0: return 0
		return self.fp / (self.fp + self.tn)


	@property
	def eval_string(self):
		e = ""
		e += "\nStats:"
		e += "\n"
		e += "\nf_{%s}:    %12s" % (np.round(0.5,8), np.round(self.f_beta(0.5),8))
		e += "\nf1:        %12s" % np.round(self.f1,8)
		e += "\nprecision: %12s" % np.round(self.precision,8)
		e += "\nrecall:    %12s" % np.round(self.recall,8)
		e += "\naccuracy:  %12s" % np.round(self.accuracy,8)
		e += "\nfalseposrate:  %12s" % np.round(self.false_positive_rate,8)
		e += "\n"
		e += "\ntp %s fn %s" % (self.tp, self.fn)
		e += "\nfp %s tn %s" % (self.fp, self.tn)
		e += "\nSum: %s" % (self.tp + self.fn + self.fp + self.tn)
		e += "\n"
		e += "\n\\perc{%12s}" % np.round(self.f_beta(0.5),8)
		e += "\n\\perc{%12s}" % np.round(self.f1,8)
		e += "\n\\perc{%12s}" % np.round(self.precision,8)
		e += "\n\\perc{%12s}" % np.round(self.recall,8)
		e += "\n\\bscm{%s}{%s}{%s}{%s}" % (self.tp, self.fn, self.fp, self.tn)
		return e

	def __str__(self):
		return self.eval_string

	def __add__(self, other):
		if other == 0: return self
		return Score(
			self.tp + other.tp,
			self.fn + other.fn,
			self.tn + other.tn,
			self.fp + other.fp
		)

	def __radd__(self, other): # for sum function
		return self.__add__(other)


##
## Iterates over a list with multiple threads.
##
## :param      args:            The arguments
## :type       args:            { type_description }
## :param      test_data_size:  The test data size
## :type       test_data_size:  { type_description }
## :param      thread_count:    The thread count
## :type       thread_count:    { type_description }
## :param      callback:        The callback that returns a result for the dataset slice
## :type       callback:        Function
##
## :returns:   list of result of thread
## :rtype:     list
##
def iterate_multi_threaded(test_data_size, thread_count, callback):

	# Multithreading

	pbar_lock = threading.Lock()
	starting_threads_pbar = tqdm(total=thread_count, desc="starting threads")
	pbar = tqdm(total=test_data_size, desc="Evaluating model")
	finished_threads_pbar = tqdm(total=thread_count, desc="finished threads")

	def startThread(dataset_slice, thread_index):
		# print("loaded classifier for thread %s" % thread_index)
		with pbar_lock: starting_threads_pbar.update()

		def step(): # to be called after processing a sample
			with pbar_lock: pbar.update()

		ret = callback(dataset_slice, step)

		with pbar_lock: finished_threads_pbar.update()
		return ret


	futures = []
	with concurrent.futures.ThreadPoolExecutor() as executor:
		for thread_index in range(thread_count):
			# print("create and start thread %s" % thread_index)
			t_from = int(test_data_size/thread_count * thread_index)
			t_to   = int(test_data_size/thread_count * (thread_index+1))
			# print("(t_from, t_to) = (%s, %s)" % (t_from, t_to))
			dataset_slice = slice(t_from, t_to)
			futures.append(executor.submit(
				startThread, dataset_slice, thread_index
			))

	pbar.close()
	starting_threads_pbar.close()

	results = [f.result() for f in futures]
	return results


# datapack stuff
# TODO refactor

MESSAGE_DELIMITER = " "

def contentToString(content):
	string = ""
	for ct in content:
		if not isNonemptyMsg(ct) or ct["type"] != "message": continue
		if string != "": string += MESSAGE_DELIMITER
		string += ct["body"]
	return string


def getSegments(chat):
	currentSegmentStart = 0
	for i, ct in enumerate(chat["content"] + [{"type": "separator"}]):
		if ct is None or ct["type"] != "separator": continue # await separator

		segment = chat["content"][currentSegmentStart:i]
		yield segment
		currentSegmentStart = i+1

def getWarningLatency(nonempty_messages, skepticism):
	mc = MasterClassifier(skepticism)
	for i, msg in enumerate(nonempty_messages):
		if "prediction" not in msg:
			# print("skipped %s" % msg)
			continue
		mc_raised_warning = mc.add_prediction(msg["prediction"] >= args.threshold)
		if mc_raised_warning: return i+1 # start at 1

PANC_MIN_MSG_NUM = 6
PANC_MAX_MSG_NUM = 150

# TODO refactor
def isGood(segment, dataset):
	"""which segments should be filtered"""
	if dataset == "PANC":
		# Filters segments with more than 150 nonempty messages.
		# The few unbehaved PAN12 segments with >150 messages are already
		# filtered in PANC/create_datapack.py, the negative segments with non
		# nonempty messages are as well. This code is only for the segments from
		# the complete positive chats originally from CC2.
		numOfNonemptyMessages = sum([isNonemptyMsg(msg) for msg in segment])
		return PANC_MIN_MSG_NUM <= numOfNonemptyMessages <= PANC_MAX_MSG_NUM
	if dataset == "VTPAN": return True # don't filter these
	return True # for other datasets also allow all segments

def isNonemptyMsg(msg):
	"""if a piece of content is a message and is nonempty"""
	return msg is not None and msg["type"] == "message" and bool(msg["body"])


def breakUpChat(chat, args, eval_mode=None):
	dataset_indicator = "PANC"
	"""if we are evaluating in segment mode, break up chat into smaller chats"""
	if not eval_mode: eval_mode = args.eval_mode

	if eval_mode.startswith("segments"):
		# only keep good segments
		return filter(lambda s: isGood(s, dataset_indicator),
		              getSegments(chat))
	# if we are running in full mode, just iterate once over the whole chat
	elif eval_mode.startswith("full"):
		return [chat["content"]]

# class EWS:
# 	def __init__(self, segment_classifier, master_classifier, window_size=50):
# 		self.segment_classifier = segment_classifier
# 		self.master_classifier = master_classifier
# 		self.window_size = window_size

# 	def processChat(self, content):
# 		window = content[:-self.window_size]
# 		prediction = self.segment_classifier.predict(contentToString(window))
# 		return self.master_classifier.add_prediction(prediction == "predator")


class MasterClassifier:

	def __init__(self, skepticism_level, last_predictions_len=10):
		# how many of the last 10 predictions have to be positive for the
		# master classifier to raise a warning
		self.skepticism_level = skepticism_level

		self.last_predictions = [False] * last_predictions_len

	##
	## Adds the latest prediction.
	##
	## :param      dangerous:  Whether the chat is deemed dangerous or not
	## :type       dangerous:  bool
	##
	## :returns:   whether the classifier raises a warning
	## :rtype:     bool
	##
	def add_prediction(self, is_dangerous):
		if self.state_is_dangerous(): return True # we stick with our choice
		self.last_predictions.insert(0, is_dangerous)
		self.last_predictions.pop() # drop oldest prediction
		return self.state_is_dangerous()

	def state_is_dangerous(self):
		return sum(self.last_predictions) >= self.skepticism_level


def get_speed(latencies, penalty_factor):
	# penalty_factor should be set such that
	# penalty(median chat length) = .5

	def penalty(delay):
		return -1 + 2 / (1 + np.exp(-penalty_factor * (delay - 1)))

	return 1 - np.median([penalty(delay) for delay in latencies])
