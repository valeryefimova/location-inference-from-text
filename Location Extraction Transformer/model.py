from transformers import BertModel
from torch import nn

class LET(nn.Module):
    def __init__(self):
        super(LET, self).__init__()
        # Base model
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True, )

        # Layer for logistic regression
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, input_mask=None, token_type_ids=None):
        output = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        # Get last hidden state from base output
        output = output[0]

        y_pred = self.linear(output)
        return y_pred
