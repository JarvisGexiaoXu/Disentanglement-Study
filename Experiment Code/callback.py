# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Base class used to build new callbacks.

"""

from typing import Any, Dict, List, Optional, Type

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64

from torch.utils.data import DataLoader
from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData, Shapes3dData, DSpritesData

from disent.dataset.transform import ToImgTensorF32
from disent.util import is_test_run  # you can ignore and remove this
from disent.metrics import metric_dci, metric_mig, metric_sap 
import wandb




class betaControlCallback(pl.Callback):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """
    # tune this method to have control over beta value during the training process.
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "BetaVae") -> None:
        """Called when the train batch ends."""
        if trainer.current_epoch == 0:
            print("callback activiated successfully....")
        
        if trainer.current_epoch == 75: # edit here to contorl the timing.
            print("trainer.current_epoch is now", trainer.current_epoch)
            print("beta was,", pl_module.cfg.beta)
            if pl_module.cfg.beta == 1:
                pl_module.cfg.beta = 10
                print("beta is now,", pl_module.cfg.beta)
#             elif pl_module.cfg.beta == 10:
#                 pl_module.cfg.beta = 1
#                 print("beta is now,", pl_module.cfg.beta)
                
        wandb.log({"beta": pl_module.cfg.beta})
        # self.beta = pl_module.cfg.beta * 1.1
        # pl_module.cfg.beta = self.beta
        if (trainer.current_epoch%5 == 4 ) or (trainer.current_epoch == 0) :#trainer.current_epoch%10 == 9:
            
            # data = XYObjectData()
            data = Shapes3dData()
            dataset = DisentDataset(data, transform=ToImgTensorF32())
            # dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
            
            print("at ",trainer.current_epoch, ", evaluating....")
            get_repr = lambda x: pl_module.encode(x.to(pl_module.device))
            a_results = {
                **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, boost_mode='sklearn'),
                **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
                **metric_sap(dataset, get_repr),
            }
            print(a_results)
            wandb.log(a_results)

        
    # remind the training has stop and report the final beta.
    def on_train_end(self, trainer, pl_module: "BetaVae"):
        print('\n Train end and beta NOW = ',pl_module.cfg.beta)
'''
    @property
    def state_key(self) -> str:
        """Identifier for the state of the callback.

        Used to store and retrieve a callback's state from the checkpoint dictionary by
        ``checkpoint["callbacks"][state_key]``. Implementations of a callback need to provide a unique state key if 1)
        the callback has state and 2) it is desired to maintain the state of multiple instances of that callback.
        """
        return self.__class__.__qualname__

    @property
    def _legacy_state_key(self) -> Type["betaControlCallback"]:
        """State key for checkpoints saved prior to version 1.5.0."""
        return type(self)

    def _generate_state_key(self, **kwargs: Any) -> str:
        """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful
        for defining a :attr:`state_key`.

        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.
        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"
'''


'''
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins."""



    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""



    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""



    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check starts."""



    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check ends."""



    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Called when the train batch begins."""

'''



'''
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""


    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""
'''