# %%

from torch.utils.data import DataLoader
from dataset.dataset_batcher import SlimDataset

# %%


def custom_collate(data):
    return data


# %%
dataset_dir = "/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch/"
bert_model_dir = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/bert_data/based_uncased"  # noqa: E501
model_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/models/slim.pt"

train_dataset = SlimDataset(root_dir=dataset_dir + "train")
train_iter = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=custom_collate)


# %%
from dataset.preprocessing import get_mini_batch
import flair
import torch

flair.device = torch.device("cpu")

from flair.embeddings import TransformerWordEmbeddings
embeddings = TransformerWordEmbeddings("bert-base-uncased",
                                       layers="-1,-2,-3,-4",
                                       fine_tune=True,
                                       layer_mean=True)

all_text_dict = {}
for i, data in enumerate(train_iter):
    mini_batches = get_mini_batch(data=data[0], size_=32)
    for b in mini_batches:
        texts = b[3]
        embeddings.embed(texts)
        embs = torch.stack([sent[0].embedding for sent in texts])
        # for text in texts:
        #     embs = []
        #     for sentence in text:
        #         embeddings.embed(sentence)
        #         embs.append(sentence[0].embedding)
        #     embs = torch.stack(embs)
        #     all_embs.append(embs)

# %%
from pympler.asizeof import asizeof

asizeof(all_text_dict) / 1024 / 1024
# %%
for i, k in all_text_dict.items():
    if len(k) != 640:
        print(i, len(k))

# %%
print(i)
# %%
