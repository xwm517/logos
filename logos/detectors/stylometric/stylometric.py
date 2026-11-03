import torch

from torch import nn
from transformers import DistilBertModel

class StylometricDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768 + 14, 1)

    def forward(self, input_ids, attention_mask, features):
        hidden = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        cls = hidden[:, 0]
        joined = torch.cat([cls, features], dim=1)

        return self.classifier(self.dropout(joined)).squeeze(-1)