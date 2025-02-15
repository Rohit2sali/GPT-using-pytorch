import torch
from typing import Dict, Union

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, text : str, characters : int, max_seq_len : int, train : bool=True):
        super(ShakespeareDataset, self).__init__()
        self.text = text
        self.characters = characters
        str_to_int_dict = {s:i for i,s in enumerate(self.characters)}
        int_to_str_dict = {i:s for i,s in enumerate(self.characters)}
        self.encoder = lambda s: [str_to_int_dict[c] for c in s]
        self.decoder = lambda l: ''.join([int_to_str_dict[i] for i in l])
        self.data = torch.tensor(self.encoder(self.text), dtype=torch.long)
        self.max_seq_len = max_seq_len
        self.train = train

    def __getitem__(self, index : int) -> Dict[str, Union[torch.Tensor, str]]:
        idx = index 
        if self.train:
            idx = torch.randint(len(self.data) - self.max_seq_len, size=(1,))
        
        x = self.data[idx : idx+self.max_seq_len]
        y = self.data[idx+1 : idx+self.max_seq_len+1]

        text = self.text[idx : idx + self.max_seq_len]
        sample = {"x" : x, "y" : y, "text" : text}
        return sample
    
    def __len__(self) -> int:
        if self.train:
            return 5000
        return len(self.data) - self.max_seq_len
