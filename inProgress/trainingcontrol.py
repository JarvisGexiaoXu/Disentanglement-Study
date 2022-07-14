import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.dataset.transform import ToImgTensorF32
from disent.util import is_test_run  # you can ignore and remove this
from disent.metrics import metric_dci, metric_mig, metric_factor_vae, metric_sap 
import trainingcallback as tc

# for callbacks use only
from typing import Any, Dict, List, Optional, Type

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

# prepare the data      length: 75000
data = XYObjectData()
dataset = DisentDataset(data, transform=ToImgTensorF32())
trainset = []
dataloader = []
for i in range(14):
    trainset.append(Subset(dataset,range(i*1000, (i+1)*1000)))
    dataloader.append(DataLoader(dataset=trainset[i], batch_size=1, shuffle=True)) 

print(len(dataloader[1]))




model=AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    )


# train the model
cfg=BetaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), loss_reduction='mean_sum', beta=1)
module: pl.LightningModule = BetaVae(model, cfg)
print('beta = ', module.cfg.beta)
#print(type(module))
trainer = pl.Trainer(logger=False, callbacks=[tc.betaControlCallback()],checkpoint_callback=False, max_steps=3000, fast_dev_run=is_test_run())

beta = 4
#print(len(dataloader))
module: pl.LightningModule = BetaVae(module._model, cfg)
#trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=1000, fast_dev_run=is_test_run())
trainer.fit(module, dataloader[i])
get_repr = lambda x: module.encode(x.to(module.device))

a_results = {
        **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, boost_mode='sklearn'),
        **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
        **metric_sap(dataset, get_repr),
    }
print("eval", i+1, ":")
print(a_results) 

# TODO 把训练过程写成loop


