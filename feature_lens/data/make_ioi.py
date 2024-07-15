from feature_lens.core.types import Model
from feature_lens.data.string_handler import StringHandler


def make_ioi(model: Model) -> StringHandler:
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    name_pairs = [
        (" John", " Mary"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]

    clean_prompts = [
        prompt.format(name)
        for (prompt, names) in zip(prompt_format, name_pairs)
        for name in names[::-1]
    ]
    corrupt_prompts = [
        prompt.format(name)
        for (prompt, names) in zip(prompt_format, name_pairs)
        for name in names
    ]
    correct_answers = [name for names in name_pairs for name in names]
    wrong_answers = [name for names in name_pairs for name in names[::-1]]

    return StringHandler(
        model=model,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        answers=correct_answers,
        wrong_answers=wrong_answers,
    )
