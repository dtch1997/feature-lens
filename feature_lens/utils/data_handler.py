"""Utilities for loading paired data."""

import abc
import eindex

from torch import Tensor
from jaxtyping import Float
from typing import Literal

from feature_lens.core.types import Model, Logits, Metric

InputType = Literal["clean", "corrupt"]


class DataHandler(abc.ABC):
    """A class that defines how to handle a batch of data."""

    @abc.abstractmethod
    def get_logits(self, model: Model, input: InputType = "clean") -> Logits:
        """Get model logits by running the model on the batch."""
        pass

    @abc.abstractmethod
    def get_metric(self, logits: Logits) -> Metric:
        """Get the metric value by comparing the logits."""
        pass

    @abc.abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abc.abstractmethod
    def get_n_pos(self) -> int:
        pass


def logits_to_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    correct_answer_tokens: Float[Tensor, " batch"],
    wrong_answer_tokens: Float[Tensor, " batch"],
) -> Logits:
    """Returns logit difference between the correct and incorrect answer."""
    last_token_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    correct_logits = eindex.eindex(
        last_token_logits, correct_answer_tokens, "batch [batch]"
    )
    incorrect_logits = eindex.eindex(
        last_token_logits, wrong_answer_tokens, "batch [batch]"
    )

    logit_diff: Float[Tensor, " batch"] = correct_logits - incorrect_logits
    return logit_diff


class SinglePromptHandler(DataHandler):
    def __init__(
        self,
        model: Model,
        clean_prompt: str,
        corrupt_prompt: str,
        answer: str,
        wrong_answer: str,
    ):
        self.clean_prompt = clean_prompt
        self.corrupt_prompt = corrupt_prompt
        self.answer = answer
        self.wrong_answer = wrong_answer

        self.tokenize(model)
        if self.clean_tokens.shape[0] != 1:
            raise ValueError(
                f"Expected batch size to be 1; got {self.clean_tokens.shape[0]}."
            )

        if self.clean_tokens.shape[1] != self.corrupt_tokens.shape[1]:
            raise ValueError(
                f"Expected the number of tokens to be the same for clean and corrupt prompts; got {self.clean_tokens.shape[1]} and {self.corrupt_tokens.shape[1]}."
            )

    def __repr__(self):
        # Multi line formatting
        return (
            f"SinglePromptHandler(\n"
            f"    clean_prompt='{self.clean_prompt}'\n"
            f"    corrupt_prompt='{self.corrupt_prompt}'\n"
            f"    answer='{self.answer}'\n"
            f"    wrong_answer='{self.wrong_answer}'\n"
            f")"
        )

    def tokenize(self, model: Model):
        self.clean_tokens = model.to_tokens(self.clean_prompt)
        self.corrupt_tokens = model.to_tokens(self.corrupt_prompt)
        self.answer_tokens = model.to_tokens(self.answer, prepend_bos=False).squeeze(0)
        self.wrong_answer_tokens = model.to_tokens(
            self.wrong_answer, prepend_bos=False
        ).squeeze(0)

        self.clean_str_tokens = model.to_str_tokens(self.clean_tokens)
        self.corrupt_str_tokens = model.to_str_tokens(self.corrupt_tokens)
        self.answer_str_tokens = model.to_str_tokens(self.answer_tokens)
        self.wrong_answer_str_tokens = model.to_str_tokens(self.wrong_answer_tokens)

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
        return 1

    def get_n_pos(self) -> int:
        return self.clean_tokens.shape[1]
