import abc
import torch

from eindex import eindex
from jaxtyping import Float
from torch import Tensor

from feature_lens.data.handler import DataHandler
from feature_lens.core.types import Logits, Metric, SingleTokens


class MetricFunction(abc.ABC):
    """A class that defines how to compute a metric."""

    @abc.abstractmethod
    def compute_metric(
        self, logits: Logits, data_handler: DataHandler, **kwargs
    ) -> Metric:
        """Compute the metric using the model and data handler."""
        pass

    def __call__(self, logits: Logits, data_handler: DataHandler, **kwargs) -> Metric:
        return self.compute_metric(logits, data_handler, **kwargs)


class LogitDiff(MetricFunction):
    """Computes the logit difference between the correct and incorrect answer."""

    def compute_metric(
        self, logits: Logits, data_handler: DataHandler, **kwargs
    ) -> Metric:
        """Returns logit difference between the correct and incorrect answer."""

        correct_answer_tokens: SingleTokens = data_handler.correct_answer_tokens
        wrong_answer_tokens: SingleTokens = data_handler.wrong_answer_tokens

        last_token_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
        correct_logits = eindex(
            last_token_logits, correct_answer_tokens, "batch [batch]"
        )
        incorrect_logits = eindex(
            last_token_logits, wrong_answer_tokens, "batch [batch]"
        )

        logit_diff: Float[Tensor, " batch"] = correct_logits - incorrect_logits
        return logit_diff


class CrossEntropy(MetricFunction):
    """Computes the cross-entropy across the clean tokens."""

    def compute_metric(
        self, logits: Logits, data_handler: DataHandler, **kwargs
    ) -> Metric:
        clean_prompt_tokens = data_handler.clean_prompt_tokens
        correct_answer_tokens = data_handler.correct_answer_tokens
        combined_tokens = torch.cat(
            (clean_prompt_tokens, correct_answer_tokens.unsqueeze(1)), dim=1
        )
        target_tokens = combined_tokens[:, 1:]

        logprobs = logits.log_softmax(dim=-1)
        selected_logprobs = eindex(logprobs, target_tokens, "batch seq [batch seq]")
        return -selected_logprobs.sum(dim=1).mean(dim=0)
