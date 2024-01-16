import os
import re
import json
import string
import numpy as np
import torch
import pickle

import tensorflow as tf
from tensorflow import keras as K

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel


### if have TPU, tranning wwith tpu
# Detect hardware, return appropriate distribution strategy.
# You can see that it is pretty easy to set up.
try:
    # TPU detection: no parameters necessary if TPU_NAME environment
    # variable is set (always set in Kaggle)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

DATASET_URL = '' #path for dataset train
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 384 #limit for len in your pargaae

# Set language name to save model
LANGUAGE = 'Vietnam'

# Depends on whether we are using TPUs or not, increase BATCH_SIZE
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

# Detect environment
ARTIFACTS_PATH = 'artifacts/'
# Import tokenizer from HuggingFace
slow_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

save_path = '%s%s-%s/' % (ARTIFACTS_PATH, LANGUAGE, MODEL_NAME)
if not os.path.exists(save_path):
    os.makedirs(save_path)

slow_tokenizer.save_pretrained(save_path)

# You can already use the Slow Tokenizer, but its implementation in Rust is much faster.
tokenizer = BertWordPieceTokenizer('%s/vocab.txt' % save_path, lowercase=True)

#set data prepare traning
class Visquad_1:
    #function init for init data format 
    def __init__(
        self,
        question,
        context,
        start_char_idx,
        answer_text,
        all_answers,
        tokenizer
    ):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.tokenizer = tokenizer
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Fix white spaces
        context = re.sub(r"\s+", ' ', context).strip()
        question = re.sub(r"\s+", ' ', question).strip()
        answer = re.sub(r"\s+", ' ', answer_text).strip()

        # Find end token index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Encode context (token IDs, mask and token types)
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Encode question (token IDs, mask and token types)
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = MAX_LEN - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets

def create_squad_examples(raw_data, tokenizer):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = Visquad_1(
                    question,
                    context,
                    start_char_idx,
                    answer_text,
                    all_answers,
                    tokenizer
                )
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y
import json

local_dataset_path = '/content/train_ViQuAD.json'

# Check if the file exists
if not os.path.exists(local_dataset_path):
    print(f"Dataset file does not exist at: {local_dataset_path}")
else:
    # Load the JSON data from the local file
    with open(local_dataset_path, 'r', encoding='utf-8') as file:
        dataset_path = json.load(file)

    # Now 'dataset' contains your data, and you can work with it as needed
    print("Dataset loaded successfully.")
raw_train_data = {}
raw_eval_data = {}
raw_train_data['data'], raw_eval_data['data'] = np.split(np.asarray(dataset_path['data']), [int(.8*len(dataset_path['data']))])
train_squad_examples = create_squad_examples(raw_train_data, tokenizer)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_squad_examples(raw_eval_data, tokenizer)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")