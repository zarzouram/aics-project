# %%

# import random
# from torch import Tensor
# from typing import Tuple
import gzip
import pathlib
from glob import iglob
import torch
from torch.utils.data import DataLoader
from dataset.dataset_batcher import SlimDataset

# %%
train_dir = ["/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch/train/"]
mytrain_dir = pathlib.Path(train_dir[0]).expanduser()
vocab_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/vocab.json"
train_files_path = iglob(str(mytrain_dir / "*.pt.gz"))

total_samples_num = 0
i = 0
for file_path in train_files_path:
    with gzip.open(file_path, "rb") as f:
        dataset = torch.load(f)
    total_samples_num += dataset[0].size(0)
    i += 1
    if i == 57:
        print(dataset[0].size(0))

print(i)
print(total_samples_num)

# %%
train_files = ["/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch/train"]
vocab_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/bert_data/based_uncased"
m = SlimDataset(roots_dir=train_files,
                bert_model_path=vocab_path,
                save_tokens_path="",
                minibatch_size=32,
                train=True)
# %%
t_iter = DataLoader(m, batch_size=1, shuffle=True, num_workers=4)

# %%
for i in t_iter:
    print(len(i))

# %%

texts1 = [
    "bank", "The river bank was flooded.", "The bank vault was robust.",
    "He had to bank on her for support."
]

texts2 = ["The bank was out of money.", "The bank teller was a man."]

# %%
from transformers import AutoTokenizer
from typing import List, Dict


class bert_preprocessing:
    def __init__(self, init_files_dir=""):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        if init_files_dir:
            self.load_status(load_dir=init_files_dir)

    def tokenize(self, sentences: List[str]) -> Dict:
        encoded_inputs = self.tokenizer(sentences,
                                        padding='longest',
                                        return_special_tokens_mask=True,
                                        return_tensors="pt")
        return encoded_inputs

    def save_status(self, save_dir: str):
        self.tokenizer.save_pretrained(save_dir)

    def load_status(self, load_dir: str):
        self.tokenizer.from_pretrained(load_dir)


# %%
x = bert_preprocessing()
y = x.tokenize(texts1)
print(y)
x.save_status("/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data")

# %%
x1 = bert_preprocessing("/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data")
test = 1
# %%
