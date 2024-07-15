import pandas as pd

from typing import Literal
from feature_lens.core.types import Model, Logits, Metric
from feature_lens.utils.data_handler import DataHandler, logits_to_logit_diff


class StringHandler(DataHandler):
    """Handles a batch of strings."""

    def __init__(
        self,
        model: Model,
        clean_prompts: list[str],
        corrupt_prompts: list[str],
        answers: list[str],
        wrong_answers: list[str],
    ):
        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts
        self.answers = answers
        self.wrong_answers = wrong_answers

        self.tokenize(model)

        if self.clean_tokens.shape[1] != self.corrupt_tokens.shape[1]:
            raise ValueError(
                f"Expected the number of tokens to be the same for clean and corrupt prompts; got {self.clean_tokens.shape[1]} and {self.corrupt_tokens.shape[1]}."
            )

    def tokenize(self, model: Model):
        self.clean_tokens = model.to_tokens(self.clean_prompts)
        self.corrupt_tokens = model.to_tokens(self.corrupt_prompts)
        self.answer_tokens = model.to_tokens(self.answers, prepend_bos=False).squeeze(
            -1
        )
        self.wrong_answer_tokens = model.to_tokens(
            self.wrong_answers, prepend_bos=False
        ).squeeze(-1)

        self.clean_str_tokens = model.to_str_tokens(self.clean_prompts)
        self.corrupt_str_tokens = model.to_str_tokens(self.corrupt_prompts)
        self.answer_str_tokens = model.to_str_tokens(self.answers, prepend_bos=False)
        self.wrong_answer_str_tokens = model.to_str_tokens(
            self.wrong_answers, prepend_bos=False
        )

    def get_logits(
        self, model: Model, input: Literal["clean", "corrupt"] = "clean", **kwargs
    ) -> Logits:
        if input == "clean":
            return model(self.clean_tokens, **kwargs)
        else:
            return model(self.corrupt_tokens, **kwargs)

    def get_metric(self, logits: Logits) -> Metric:
        return logits_to_logit_diff(
            logits, self.answer_tokens, self.wrong_answer_tokens
        )

    def get_batch_size(self) -> int:
        return self.clean_tokens.shape[0]

    def get_n_pos(self) -> int:
        return self.clean_tokens.shape[1]


def as_dataframe(handler: StringHandler) -> pd.DataFrame:
    """Converts a StringHandler to a pandas DataFrame."""
    data = {
        "clean_prompt": handler.clean_prompts,
        "corrupt_prompt": handler.corrupt_prompts,
        "answer": handler.answers,
        "wrong_answer": handler.wrong_answers,
    }
    return pd.DataFrame(data)
