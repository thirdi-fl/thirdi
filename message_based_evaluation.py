from eval_util import getEvalArgs, getSegments, MasterClassifier, isNonemptyMsg, Score, get_speed, breakUpChat
from pathlib import Path
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
	"--full_eval_mode",
	dest='full_eval_mode',
	help="which mode was used to annotate the full-length predator chats",
	choices=["full", "full_fast"],
	default="full_fast"
)
parser.add_argument(
	"--segments_eval_mode",
	dest='segments_eval_mode',
	help="which mode was used to annotate the segments",
	choices=["segments", "segments_fast"],
	default="segments_fast"
)

parser.add_argument(
	"--skepticism",
	dest='skepticism',
	help="skepticism level of the master classifier",
	type=int,
	default=5
)
parser.add_argument(
	"--window_size",
	dest='window_size',
	help="window_size used for prediction annotation",
	type=int,
	default=50
)
args = getEvalArgs(parser)


def getTimestamp():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def getUNIXTimestamp():
	return int(datetime.now().replace(tzinfo=timezone.utc).timestamp()*1000)

# load datapack with segment level annotations
out_dir = "/evaluation/results/%s/message_based_eval/" % (args.output_dir)
datapack_path = "/evaluation/results/%s/message_based_eval/annotated-datapack-PANC-test-eval_mode-segments_fast--window_size-50.json" % (args.output_dir)

with open(datapack_path, "r") as file:
	datapack_segments = json.load(file)
chatNames = sorted(list(datapack_segments["chats"].keys()))

# load datapack with chat level annotations
datapack_path = "/evaluation/results/%s/message_based_eval/annotated-datapack-PANC-test-eval_mode-full_fast--window_size-50.json" % (args.output_dir)
with open(datapack_path, "r") as file:
	datapack_full = json.load(file)

	

# information about datapacks can be found in the chat-visualizer repo

def getWarningLatency(nonempty_messages, skepticism):
	mc = MasterClassifier(skepticism)
	for i, msg in enumerate(nonempty_messages):
		if "prediction" not in msg:
			# print("skipped %s" % msg)
			continue
		mc_raised_warning = mc.add_prediction(msg["prediction"] >= args.threshold)
		if mc_raised_warning: return i+1 # start at 1

MAX_MESSAGE_LEN = 150
MESSAGE_WITH_HALF_PENALTY = 90
# set penalty_factor such that the penalty for a latency equal to the
# median chat length is 0.5
# enter "-1+2/(1+exp(-x(k-1))) = 0.5 solve for real x" on wolframalpha
penalty_factor = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

def evaluate_for_skepticism(skepticism):
	full_len_score = Score()
	message_scores_with_dropout = [Score() for i in range(MAX_MESSAGE_LEN)]
	latencies = []

	# get latencies for full length chats
	for chatName, chat in datapack_full["chats"].items():
		if not chat["className"] == "predator": continue
		nonempty_messages = [ct for ct in chat["content"] if isNonemptyMsg(ct)]
		latency = getWarningLatency(nonempty_messages, skepticism)
		# we only count the latency of true positives warnings
		if latency:
			latencies.append(latency)

	# from this get the speed
	speed = get_speed(latencies, penalty_factor)

	# get classification accuracy for segments
	for chatName, chat in datapack_segments["chats"].items():
		is_positive = chat["className"] == "predator"

		for segment in breakUpChat(chat, args, eval_mode="segments"):
			nonempty_messages = [ct for ct in segment if isNonemptyMsg(ct)]

			# note warning decisions per message
			mc = MasterClassifier(skepticism) # master classifier for this segment
			for i, msg in enumerate(nonempty_messages):
				score = message_scores_with_dropout[i]
				if "prediction" not in msg:
					# print("skipped %s" % msg)
					continue
				predicted_positive = msg["prediction"] >= args.threshold
				mc_raised_warning = mc.add_prediction(predicted_positive)
				score.add_prediction(mc_raised_warning, is_positive)

			# note warning decision for the whole chat
			# Once it gets dangerous, the MC keeps its state dangerous.
			full_len_score.add_prediction(mc.state_is_dangerous(), is_positive)

	# finally get f_latency
	f_latency = full_len_score.f1 * speed

	return message_scores_with_dropout, full_len_score, latencies, speed, f_latency

full_len_scores = []
speeds = []
f_latencies = []

for skepticism in tqdm(range(1, 10+1), desc="Evaluating for skepticism"):
	print("For skepticism = %s" % skepticism)
	message_scores_with_dropout, full_len_score, latencies, speed, f_latency = \
		evaluate_for_skepticism(skepticism)

	print("Score for segments at full length:")
	print(full_len_score)

	# evaluation metrics for skepticism
	f1          = [score.f1            for score in message_scores_with_dropout]
	precision   = [score.precision     for score in message_scores_with_dropout]
	recall      = [score.recall        for score in message_scores_with_dropout]
	false_positive_rate = [score.false_positive_rate for score in message_scores_with_dropout]
	sample_nums = [score.number_of_samples for score in message_scores_with_dropout]

	print("np.median(latencies) = %s" % np.median(latencies))
	# print(latencies)

	print("f_latency = f1 * speed = %s * %s = %s" % (full_len_score.f1, speed, f_latency))

	# save warning latencies for skepticism
	latency_dir = os.path.join(out_dir, "full_length_latencies/")
	Path(latency_dir).mkdir(parents=True, exist_ok=True)
	latencyFile = "%s/latencies__skepticism-%s.txt" % (latency_dir, skepticism)
	with open(latencyFile, "w") as file:
		for val in latencies: file.write("%s\n" % str(val))

	# save eval metrics for skepticism
	eval_dir = os.path.join(out_dir, "segment_accuracy_by_num_of_messages/skepticism-%s/" % skepticism)
	Path(eval_dir).mkdir(parents=True, exist_ok=True)
	for name, scores in ({"f1": f1, "precision": precision, "recall": recall, "samples": sample_nums}).items():
		with open("%s/%s.txt" % (eval_dir, name), "w") as file:
			# files have 100 lines: of the values of the metrics at each message_num
			for val in scores: file.write("%s\n" % str(val))

	# save for later use
	full_len_scores.append(full_len_score)
	speeds.append(speed)
	f_latencies.append(f_latency)


# save eval metrics by skepticism
f1          = [score.f1            for score in full_len_scores]
precision   = [score.precision     for score in full_len_scores]
recall      = [score.recall        for score in full_len_scores]
false_positive_rate = [score.false_positive_rate for score in full_len_scores]
eval_dir = os.path.join(out_dir, "metrics_by_skepticism/")
Path(eval_dir).mkdir(parents=True, exist_ok=True)
for name, scores in ({"f1": f1, "precision": precision, "recall": recall, "fpr": false_positive_rate ,"speed": speeds, "f_latency": f_latencies}).items():
	with open("%s/%s.txt" % (eval_dir, name), "w") as file:
		# files have 100 lines: of the values of the metrics at each message_num
		for val in scores: file.write("%s\n" % str(val))
 