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
SIMILARITY = "semantic_similarity"
TARGET_LABEL_PROB = "target_label_probability_from_victim"
PROMPT_ORIGINAL_TARGET_LABEL_PROB = "prompt_original_target_label_probability_from_victim"


# Miscellaneous
PLOTTING_MOVING_AVERAGE_WINDOW_LENGTH = 128
MODEL_RESPONSE = "model_response"

NEGATIVE_LABEL = "negative"
POSITIVE_LABEL = "positive"
LABEL_NAME_TO_CODE = {
    NEGATIVE_LABEL: 0,
    POSITIVE_LABEL: 1,
}
