import argparse
import torch
import numpy as np
import pandas as pd

from pathlib import Path

from embedding import vocabulary_from_texts, VocabularyEmbedding
from model import LIT
from utils import tokenize_text, detokenize_word

LAUNCH_DIR = Path().absolute()

DATA_DIR = "data"
DATASET_FILE = "COCO-locations.csv"

MODELS_DIR = "models"
WEIGHTS_FILE = "bert-location-inference-transformer-epoch-last.pt"

VOCABULARY_SIZE = 7453
EMBEDDING_SIZE = 256


def initialize_model(verbose=False):
    path_to_data = LAUNCH_DIR.joinpath(DATA_DIR, DATASET_FILE)

    coco_data = pd.read_csv(path_to_data)
    coco_texts = list(coco_data['cap'])
    coco_backgrounds = list(coco_data['background'])

    vocabulary = vocabulary_from_texts(coco_texts + coco_backgrounds)
    embedding = VocabularyEmbedding(vocabulary, EMBEDDING_SIZE)

    if verbose:
        print("Vocabulary: COCO vocabulary loaded")

    model = LIT(VOCABULARY_SIZE, EMBEDDING_SIZE)

    path_to_model = LAUNCH_DIR.joinpath(MODELS_DIR, WEIGHTS_FILE)
    if not Path(path_to_model).is_file():
        print(path_to_model)
        raise FileNotFoundError("Model: weights were not loaded")

    model.load_state_dict(torch.load(path_to_model))

    if verbose:
        print("Model: last epoch loaded")

    return {"model": model, "embedding": embedding}


def extract_location(text, candidate, verbose=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Device: cuda selected")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Device: cpu selected")

    load = initialize_model(verbose=verbose)

    embedding = load["embedding"]

    if candidate not in embedding.word_to_ix:
        raise ValueError("Candidate not found in vocabulary")

    embed_candidate = embedding.word_to_ix[candidate]

    model = load['model']
    model.to(device)

    model.eval()

    tokenized_text = tokenize_text(text)
    input_ids = tokenized_text['input_ids']
    attention_masks = tokenized_text['attention_masks']

    if verbose:
        print("Text: tokenized")

    texts_count = len(input_ids)

    with torch.no_grad():
        b_input_ids = input_ids.to(device)
        b_attention_masks = attention_masks.to(device)
        b_candidates = torch.tensor([embed_candidate] * texts_count).to(device)

        logits = model(b_input_ids, b_candidates, input_mask=b_attention_masks).squeeze()
        logits = logits.detach().cpu().numpy()
        is_relevant = False

        for logit in np.atleast_2d(logits):
            is_relevant = is_relevant or logit > 0

        return is_relevant

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine how the location relates to the text")

    required = parser.add_argument_group()
    required.add_argument("-raw_text", help="required raw text to process")
    required.add_argument("-location", help="required location to check if it is related to text as location")
    not_required = parser.add_argument_group()
    not_required.add_argument("-verbose", help="pass True if you need log output")

    args = parser.parse_args()
    raw_text = args.raw_text
    location = args.location

    if raw_text is None or location is None:
        raise ValueError("Text and location are required to process")

    verbose = args.verbose == "True"

    text = raw_text
    print("Inferening location from text: {}".format(raw_text))
    print("Candidate: {}".format(location))

    is_relevant = extract_location(raw_text, location, verbose=verbose)

    if is_relevant:
        print("Location: {} is relevant location to the text".format(location))
    else:
        print("Location: {} is not relevant location to the text".format(location))
