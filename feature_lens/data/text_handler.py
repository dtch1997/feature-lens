import eindex

from typing import Literal
from feature_lens.core.types import Model, Logits, Metric


class TextHandler:
    """Handles a single, unpaired text string"""

    def __init__(self, model: Model, text: str):
        self.text = text
        self.tokenize(model)

    def get_logits(
        self, model: Model, input: Literal["clean", "corrupt"] = "clean", **kwargs
    ) -> Logits:
        if input != "clean":
            raise ValueError(
                "TextHandler only supports clean input as it is unpaired. "
            )
        return model(self.curr_tokens, **kwargs)

    def get_metric(self, logits: Logits) -> Metric:
        # Return cross-entropy loss from logits
        logprobs = logits.log_softmax(dim=-1)
        selected_logprobs = eindex.eindex(
            logprobs, self.next_tokens, "batch seq [batch seq]"
        )
        return -selected_logprobs.sum(dim=1).mean(dim=0)

    def tokenize(self, model: Model):
        text_tokens = model.to_tokens(self.text)
        self.curr_tokens = text_tokens[:, :-1]
        self.next_tokens = text_tokens[:, 1:]
