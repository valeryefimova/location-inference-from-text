from transformers import BertModel
from torch import bmm, nn


class LIT(nn.Module):
    def __init__(self, vocabulary_length, hidden_dim):
        super(LIT, self).__init__()
        # Base model
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Bert parameters freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # Embedding
        self.embedding = nn.Embedding(vocabulary_length, hidden_dim)
        # Layer for logistic regression
        self.linear = nn.Linear(768, hidden_dim)

    def forward(self, input_ids, input_background, input_mask=None, token_type_ids=None):
        # CLS token
        output = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        output = output['pooler_output']

        output = self.linear(output).unsqueeze(1)

        # Location embedding
        background = self.embedding(input_background).unsqueeze(2)

        # Scalar product
        y_pred = bmm(output, background)
        return y_pred
