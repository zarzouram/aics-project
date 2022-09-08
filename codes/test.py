# %%
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from helpers.scheduler import LinearDecayLR, XfmrWarmupScheduler

# %%
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = Adam(model, lr=1)
lr_scheduler2 = LinearDecayLR(optimizer=optimizer)


# %%
lrs = {}
for epoch in range(1, int(1e6) + 1):
    optimizer.step()
    lr_scheduler2.step()
    lr = optimizer.param_groups[0]["lr"]
    lrs[epoch] = lr

# %%
plt.figure(figsize=(8, 3))
plt.plot(list(lrs.keys()), list(lrs.values()))
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.show()

# %%
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = Adam(model, lr=1)
lr_scheduler1 = XfmrWarmupScheduler(optimizer=optimizer,
                                    warmup=8000,
                                    step_num=1e6)

# %%
lrs = {}
for epoch in range(1, int(1e6) + 1):
    optimizer.step()
    lr_scheduler1.step()
    lr = optimizer.param_groups[0]["lr"]
    lrs[epoch] = lr

# %%
plt.figure(figsize=(8, 3))
plt.plot(list(lrs.keys()), list(lrs.values()))
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.show()
