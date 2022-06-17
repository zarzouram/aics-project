# %%
import torch
from dataset.dataset_batcher import SlimDataset
from torch.utils.data import DataLoader
from dataset.preprocessing import get_mini_batch

# %%
dataset_dir = "/scratch/guszarzmo/aicsproject/data/slim/turk_data_torch/"

# %%
train_dataset = SlimDataset(root_dir=dataset_dir + "train")
train_iter = DataLoader(train_dataset, batch_size=1, shuffle=False)

# %%
vs = []
for train_batch in train_iter:
    trn_mini_b = get_mini_batch(data=train_batch, size_=0)
    views = torch.cat((trn_mini_b[2], trn_mini_b[1].unsqueeze(1)), dim=1)
    b = views.size(-1)
    vs.append(views.view(-1 , b))

# %%
torch.cat(vs, dim=0).size()
# %%
torch.unique(torch.cat(vs, dim=0)[:, :2], dim=0)
# %%
torch.max(torch.cat(vs, dim=0), dim=0)

# %%
torch.min(torch.cat(vs, dim=0), dim=0)
# %%
torch.unique(torch.round(torch.cat(vs, dim=0), decimals=4), dim=0).size()
# %%
torch.cat(vs, dim=0).size(0)
# %%
