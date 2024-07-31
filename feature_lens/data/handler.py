"""Utilities for loading data."""

from __future__ import annotations

import pandas as pd

from typing import Literal
from dataclasses import dataclass
from feature_lens.core.types import Model, Logits, Tokens, SingleTokens

InputType = Literal["clean", "corrupt"]


@dataclass(frozen=True)
class Dataset:
    """Contains the original strings for the paired data."""

    clean_prompts: list[str]
    corrupt_prompts: list[str]
    answers: list[str]
    wrong_answers: list[str]

    def as_dataframe(self) -> pd.DataFrame:
        """Converts a StringInfo to a pandas DataFrame."""
        data = {
            "clean_prompt": self.clean_prompts,
            "corrupt_prompt": self.corrupt_prompts,
            "answer": self.answers,
            "wrong_answer": self.wrong_answers,
        }
        return pd.DataFrame(data)


@dataclass(frozen=True)
class TokenInfo:
    """Contains the tokens strings for the paired data."""

    clean_str_tokens: list[list[str]]
    corrupt_str_tokens: list[list[str]]
    answer_str_tokens: list[list[str]]
    wrong_answer_str_tokens: list[list[str]]


@dataclass(frozen=True)
class DataHandler:
    """A class that defines how to handle a batch of paired clean / corrupt tokens."""

    clean_prompt_tokens: Tokens
    corrupt_prompt_tokens: Tokens
    correct_answer_tokens: SingleTokens
    wrong_answer_tokens: SingleTokens
    str_token_info: TokenInfo

    def __post_init__(self):
        self._validate_token_shapes()

    def get_logits(self, model: Model, input: str = "clean", *args, **kwargs) -> Logits:
        """Get the logits for the input."""
        if input == "clean":
            return model(self.clean_prompt_tokens, *args, **kwargs)
        elif input == "corrupt":
            return model(self.corrupt_prompt_tokens, *args, **kwargs)
        else:
            raise ValueError(f"Invalid input type: {input}")

    def _validate_token_shapes(self):
        if not (
            self.clean_prompt_tokens.shape[0]
            == self.corrupt_prompt_tokens.shape[0]
            == self.correct_answer_tokens.shape[0]
            == self.wrong_answer_tokens.shape[0]
        ):
            raise ValueError(
                f"Expected the batch size to be the same for clean and corrupt prompts; got {self.clean_prompt_tokens.shape[0]} and {self.corrupt_prompt_tokens.shape[0]} and {self.correct_answer_tokens.shape[0]} and {self.wrong_answer_tokens.shape[0]}."
            )

        if self.clean_prompt_tokens.shape[1] != self.corrupt_prompt_tokens.shape[1]:
            raise ValueError(
                f"Expected the sequence length to be the same for clean and corrupt prompts; got {self.clean_prompt_tokens.shape[1]} and {self.corrupt_prompt_tokens.shape[1]}."
            )

    @property
    def batch_size(self) -> int:
        return self.clean_prompt_tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.clean_prompt_tokens.shape[1]


def build_handler(
    model: Model,
    dataset: Dataset,
) -> DataHandler:
    clean_prompts = dataset.clean_prompts
    corrupt_prompts = dataset.corrupt_prompts
    answers = dataset.answers
    wrong_answers = dataset.wrong_answers

    # Tokenize the strings
    clean_prompt_tokens = model.to_tokens(clean_prompts)
    corrupt_prompt_tokens = model.to_tokens(corrupt_prompts)
    correct_answer_tokens = model.to_tokens(answers, prepend_bos=False).squeeze(-1)
    wrong_answer_tokens = model.to_tokens(wrong_answers, prepend_bos=False).squeeze(-1)

    # Get the string tokens
    clean_str_tokens = model.to_str_tokens(clean_prompts)
    corrupt_str_tokens = model.to_str_tokens(corrupt_prompts)
    answer_str_tokens = model.to_str_tokens(answers, prepend_bos=False)
    wrong_answer_str_tokens = model.to_str_tokens(wrong_answers, prepend_bos=False)

    return DataHandler(
        clean_prompt_tokens=clean_prompt_tokens,
        corrupt_prompt_tokens=corrupt_prompt_tokens,
        correct_answer_tokens=correct_answer_tokens,
        wrong_answer_tokens=wrong_answer_tokens,
        str_token_info=TokenInfo(
            clean_str_tokens=clean_str_tokens,  # type: ignore
            corrupt_str_tokens=corrupt_str_tokens,  # type: ignore
            answer_str_tokens=answer_str_tokens,  # type: ignore
            wrong_answer_str_tokens=wrong_answer_str_tokens,  # type: ignore
        ),
    )


def build_unpaired_handler(
    model: Model,
    text: str,
) -> DataHandler:
    # Tokenize the strings
    all_tokens = model.to_tokens([text])
    clean_prompt_tokens = all_tokens[:, :-1]
    correct_answer_tokens = all_tokens[:, -1]

    corrupt_prompt_tokens = clean_prompt_tokens
    wrong_answer_tokens = correct_answer_tokens

    # Get the string tokens
    full_str_tokens = model.to_str_tokens([text])
    clean_str_tokens = [t[:-1] for t in full_str_tokens]
    answer_str_tokens = [t[-1] for t in full_str_tokens]
    wrong_answer_str_tokens = answer_str_tokens
    corrupt_str_tokens = clean_str_tokens

    return DataHandler(
        clean_prompt_tokens=clean_prompt_tokens,
        corrupt_prompt_tokens=corrupt_prompt_tokens,
        correct_answer_tokens=correct_answer_tokens,
        wrong_answer_tokens=wrong_answer_tokens,
        str_token_info=TokenInfo(
            clean_str_tokens=clean_str_tokens,  # type: ignore
            corrupt_str_tokens=corrupt_str_tokens,  # type: ignore
            answer_str_tokens=answer_str_tokens,  # type: ignore
            wrong_answer_str_tokens=wrong_answer_str_tokens,  # type: ignore
        ),
    )
