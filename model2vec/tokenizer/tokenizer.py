from __future__ import annotations

import logging
import re

from skeletoken import TokenizerModel

logger = logging.getLogger(__name__)


def clean_and_create_vocabulary(
    model: TokenizerModel,
    vocabulary_to_add: list[str],
    token_remove_regex: re.Pattern[str] | None,
) -> TokenizerModel:
    """Cleans a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    seen_tokens = set()

    internal_tokens: list[str] = model.sorted_vocabulary
    if token_remove_regex:
        internal_tokens = [x for x in internal_tokens if not token_remove_regex.match(x)]
    preprocessor = model.preprocessor

    seen_tokens = set(internal_tokens)
    tokens_to_add: list[str] = []
    added_tokens_to_add: list[str] = []
    for token in vocabulary_to_add:
        preprocessed = preprocessor.preprocess(token)
        if len(preprocessed) < 1:
            logger.warning(f"Token '{token}' was empty after preprocessing.")
            continue
        if len(preprocessed) > 1:
            logger.warning(f"Token '{token}' was split into multiple tokens after preprocessing.")
            added_tokens_to_add.append(token)
            continue
        token = preprocessed[0]
        if token in seen_tokens:
            logger.warning(f"Token '{token}' was already in the vocabulary.")
            continue
        if token_remove_regex and token_remove_regex.match(token):
            logger.warning(f"Token '{token}' was removed due to regex match.")
            continue
        seen_tokens.add(token)
        tokens_to_add.append(token)

    model = model.add_tokens_to_vocabulary(tokens_to_add, preprocess_tokens=True)
    model = model.add_addedtokens(added_tokens_to_add, is_special=False, single_word=False, normalized=True)

    return model


def turn_tokens_into_ids(tokens: list[str], model: TokenizerModel) -> list[list[int]]:
    """
    Convert a list of Token objects to their corresponding token ID sequences.

    :param tokens: List of Token objects to convert
    :param model: The tokenizermodel of the tokenizer.
    :return: List of token IDs corresponding to the input tokens
    """
    prefix, suffix = model.bos_ids or [], model.eos_ids or []
    vocabulary = model.vocabulary
    tokenizer = model.to_tokenizer()

    token_ids: list[list[int]] = []
    for token in tokens:
        if token_id := vocabulary.get(token):
            token_ids.append([*prefix, token_id, *suffix])
        else:
            token_ids.append(tokenizer.encode(token).ids)

    return token_ids
