# %%
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from helpers.scheduler import LinearDecayLR, XfmrWarmupScheduler

# %%
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = Adam(model, lr=5e-4)
lr_scheduler1 = XfmrWarmupScheduler(optimizer=optimizer,
                                    warmup=8000,
                                    step_num=1e6)

lr_scheduler2 = LinearDecayLR(optimizer=optimizer)


# %%
for epoch in range(20):
    for s in range(10):
        optimizer.step()
    lr_scheduler1.step(epoch=s)

# %%
epochs = list(range(int(1e6)))
plt.figure(figsize=(8, 3))
plt.plot(epochs, [lr_scheduler1.get_lr_factor(e) for e in epochs])
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.title("Cosine Warm-up Learning Rate Scheduler")
plt.show()

# %%
print([lr_scheduler1.get_lr_factor(e) for e in epochs][-1])
print([lr_scheduler1.get_lr_factor(e) for e in epochs][20000])

# %%
epochs = list(range(int(1e6)))
plt.figure(figsize=(8, 3))
plt.plot(epochs, [lr_scheduler2.get_lr_factor(e) for e in epochs])
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.title("Cosine Warm-up Learning Rate Scheduler")
plt.show()
# %%
print([lr_scheduler2.get_lr_factor(e) for e in epochs][-1])
# %%
