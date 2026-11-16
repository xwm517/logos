import torch

from torch import nn
from transformers import DistilBertModel, get_linear_schedule_with_warmup


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


def train(model, loader, epochs=3, lr=2e-5, device='cpu'):
    model.to(device).train()

    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    steps = len(loader) * epochs
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.1 * steps), steps)

    for epoch in range(epochs):
        running = 0.0

        for batch in loader:
            optimiser.zero_grad()

            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                features=batch['features'].to(device),
            )
            loss = loss_fn(logits, batch['labels'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimiser.step()
            schedule.step()
            running += loss.item()

        print(f'epoch {epoch + 1}: loss {running / len(loader):.4f}', flush=True)

    return model


@torch.no_grad()
def score(model, loader, device='cpu'):
    # The raw logit, not sigmoid(logit). Ordering is the same, so TPR@1%FPR and
    # AUROC are unchanged, but a probability saturates: on HC3 90% of the
    # ChatGPT scores sat inside a range of 0.0005. The fusion stacks this score
    # with Binoculars in a logistic regression, and a saturated input there
    # needs an enormous coefficient and gives weights that cannot be read.
    model.to(device).eval()
    scores = []

    for batch in loader:
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            features=batch['features'].to(device),
        )
        scores.append(logits.cpu())

    return torch.cat(scores).numpy()
