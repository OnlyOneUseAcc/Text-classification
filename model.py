import torch
from transformers import RobertaModel


class TweetModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TweetModel, self).__init__()
        self.l1 = RobertaModel.from_pretrained('roberta-base')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, n_classes)

    def forward(self, ids, mask):
        out = self.l1(ids, attention_mask=mask)[1]
        out = self.l2(out)
        out = self.l3(out)
        return out
