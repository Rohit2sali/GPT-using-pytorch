import torch
from dataset import ShakespeareDataset
from typing import Tuple

def return_dataset(data_path : int, split : int, max_seq_len : int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    with open(data_path, 'r', encoding="utf-8") as f:
        text = f.read()
    text = text[:300]

    characters = sorted(list(set(text)))
    dataset_len = len(text)
    train_size = int(dataset_len * split)

    train_text = text[:train_size]
    test_text = text[train_size:]

    train_set = ShakespeareDataset(train_text, characters, max_seq_len, train=True)
    test_set = ShakespeareDataset(test_text, characters, max_seq_len, train=True)
    return train_set, test_set
