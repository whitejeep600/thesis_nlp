from enum import Enum

# SST2 and sample data names
LABEL = "label"
SENTENCE = "sentence"
ID = "idx"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
ORIGINAL_SENTENCE = "original_sentence"


# Train/dataset modes
class TrainMode(Enum):
    train = "train"
    eval = "eval"


# Reward and metric names
REWARD = "reward"
