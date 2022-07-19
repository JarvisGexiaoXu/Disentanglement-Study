import pytorch_lightning as pl
from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.frameworks.ae import Ae
from disent.frameworks.vae import BetaVae, AdaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.dataset.transform import ToImgTensorF32
from disent.util import is_test_run  # you can ignore and remove this
from disent.metrics import metric_dci, metric_mig, metric_sap 
import torch
import matplotlib.pyplot as plt
import numpy as np


def AE():
    # create the pytorch lightning system
    module: pl.LightningModule = Ae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=Ae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), loss_reduction='mean_sum')
    )
    return module

def BetaVAE():
    # create the pytorch lightning system
    module: pl.LightningModule = BetaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=BetaVae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), loss_reduction='mean_sum', beta=4)
    )
    return module

def AdaGVAE():
    # create the pytorch lightning system
    module: pl.LightningModule = AdaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        ),
        cfg=AdaVae.cfg(
            optimizer='adam', optimizer_kwargs=dict(lr=1e-3),
            loss_reduction='mean_sum', beta=4, ada_average_mode='gvae', ada_thresh_mode='kl',
        )
    )
    return module
    



# prepare the data
data = XYObjectData()
dataset = DisentDataset(data, transform=ToImgTensorF32()) 
dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

module = BetaVAE()
# train the model
trainer = pl.Trainer(logger=False, max_epochs=1, checkpoint_callback=False, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)

module.eval()

get_repr = lambda x: module.encode(x.to(module.device))

a_results = {
        **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, boost_mode='sklearn'),
        **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
        **metric_sap(dataset, get_repr),
    }
print(a_results)


# def show_image(x, idx):
#     x = x.view(128, 28, 28)

#     fig = plt.figure()
#     plt.imshow(x[idx].cpu().numpy())
# noise = torch.randn(128, 6).to(module.device)
# generated_images = module.decode(noise)
# image = generated_images.detach().numpy()[0]
# for i in range(10):
#     image = dataset[i*1000]['x_targ'][0]
#     # print(type(dataset[0]['x_targ']))
#     # print(image)
#     image = np.transpose(image,(1,2,0))
#     plt.imshow(image)
#     plt.show()