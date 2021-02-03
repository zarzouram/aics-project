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
vocab_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/vocab.json"
m = SlimDataset(roots_dir=train_files,
                vocab_path=vocab_path,
                minibatch_size=32,
                train=True)

t_iter = DataLoader(m, batch_size=1, shuffle=True, num_workers=4)

for i in t_iter:
    print(len(i))

# %%
