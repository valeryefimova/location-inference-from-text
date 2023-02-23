import argparse
import torch
import numpy as np

from pathlib import Path
from model import LET
from utils import tokenize_text, detokenize_word

LAUNCH_DIR = Path().absolute()
WEIGHTS_FILE = "bert-location-extraction-transformer-epoch-last.pt"


def initialize_model(verbose=False):
    model = LET()

    path_to_model = LAUNCH_DIR.joinpath("models", WEIGHTS_FILE)
    if not Path(path_to_model).is_file():
        print(path_to_model)
        raise FileNotFoundError("Model: weights were not loaded")

    model.load_state_dict(torch.load(path_to_model))
    if verbose:
        print("Model: last epoch loaded")
    return model


def extract_location(text, verbose=False):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Device: cuda selected")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Device: cpu selected")

    model = initialize_model(verbose=verbose)
    model.to(device)

    model.eval()

    tokenized_text = tokenize_text(text)
    input_ids = tokenized_text['input_ids']
    attention_masks = tokenized_text['attention_masks']

    if verbose:
        print("Text: tokenized")

    with torch.no_grad():
        b_input_ids = input_ids.to(device)
        b_attention_masks = attention_masks.to(device)

        logits = model(b_input_ids, b_attention_masks).squeeze()
        logits = np.atleast_2d(logits.detach().cpu().numpy())
        logits = np.array([np.array(list(map(lambda x: 1 if x > 0 else 0, l))) for l in logits])

        candidates = set()
        for (logit, text) in list(zip(logits, input_ids)):
            for i in range(len(logit)):
                if logit[i] == 1:
                    candidates.add(text[i])

        candidates = set(map(lambda c: detokenize_word(c), candidates))

        if verbose:
            print("Locations: extracted")

        return list(candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract location from text")

    required = parser.add_mutually_exclusive_group(required=True)
    required.add_argument("-raw_text", help="required raw text to process")
    not_required = parser.add_mutually_exclusive_group()
    not_required.add_argument("-verbose", help="pass True if you need log output")

    args = parser.parse_args()
    raw_text = args.raw_text
    verbose = args.verbose == "True"

    text = raw_text
    print("Extracting location from text: {}".format(raw_text))
    print("Extracted location candidates: {}".format(", ".join(extract_location(raw_text, verbose=verbose))))
