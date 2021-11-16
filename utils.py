from typing import List, Union

def get_split_len(split_fraction: Union[int, float], len_dataset: int):
    if isinstance(split_fraction, int):
        return split_fraction
    elif isinstance(split_fraction, float):
        return int(split_fraction * len_dataset)
    else:
        raise ValueError(f"Unsupported type {type(split_fraction)} for split fraction.")


def get_split_lengths(split_fractions: List[Union[int, float]], len_dataset: int):
    split_lengths = [get_split_len(fraction, len_dataset) for fraction in split_fractions]
    splits_sum = sum(split_lengths)
    if splits_sum > len_dataset:
        raise ValueError(f"Dataset length is smaller than sum of "
                         f"specified splits ({split_lengths}): {len_dataset} < {splits_sum}.")
    split_lengths.insert(0, len_dataset - splits_sum)
    return split_lengths


def int_or_float_type(x: str) -> Union[int, float]:
    if '.' in x:
        return float(x)
    else:
        return int(x)


def int_or_str_type(x: str) -> Union[int, str]:
    try:
        return int(x)
    except ValueError:
        return x
