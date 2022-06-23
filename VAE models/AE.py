import pytorch_lightning as pl
from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.frameworks.ae import Ae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64
from disent.dataset.transform import ToImgTensorF32
from disent.util import is_test_run  # you can ignore and remove this
from disent.metrics import metric_dci, metric_mig, metric_factor_vae, metric_sap 



# prepare the data
data = XYObjectData()
dataset = DisentDataset(data, transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# create the pytorch lightning system
module: pl.LightningModule = Ae(
    model=AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=6),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
    ),
    cfg=Ae.cfg(optimizer='adam', optimizer_kwargs=dict(lr=1e-3), loss_reduction='mean_sum')
)

# train the model
trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)

get_repr = lambda x: module.encode(x.to(module.device))

a_results = {
        **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, boost_mode='sklearn'),
        **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
        **metric_sap(dataset, get_repr),
    }
print(a_results)  