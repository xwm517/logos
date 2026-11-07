import torch

from torch.utils.data import Dataset


class StylometricDataset(Dataset):
    def __init__(self, texts, labels, features, tokenizer, max_length=256):
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

        assert len(self.input_ids) == len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'features': self.features[index],
            'labels': self.labels[index],
        }