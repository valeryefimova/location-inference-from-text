from nltk import tokenize
from torch import cat
from string import punctuation
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_TEXT_SIZE = 256


def tokenize_text(text):
    texts = []
    sentences = tokenize.sent_tokenize(text)
    for sentence in sentences:
        lowered_sentence = (sentence.translate(str.maketrans('', '', punctuation))).lower()
        texts.append(lowered_sentence)

    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_TEXT_SIZE,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    return {'input_ids': cat(input_ids, dim=0), 'attention_masks': cat(attention_masks, dim=0)}


def detokenize_word(input_id):
    return "".join(tokenizer.decode(input_id).split())
