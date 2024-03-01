import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import Subset
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, AveragePrecision
from torchmetrics.collections import MetricCollection
import torch.optim as optim
import numpy as np
import ankh

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        # print('Current lr: ', [base_lr * lr_factor for base_lr in self.base_lrs])
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1) * 1.0 / self.warmup
        return lr_factor


class BaselineModel(pl.LightningModule):
    def __init__(self, params):
        super(BaselineModel, self).__init__()

        self.save_hyperparameters(params)

        # Transfer to hyperparameters
        self.train_set = None
        self.val_set = None
        self.test_set = None

        # Defining whether to sync the logs or not depending on the number of gpus
        if hasattr(self.hparams, 'devices') and int(self.hparams.devices) > 1:
            self.hparams.sync_dist = True
        else:
            self.hparams.sync_dist = False

        self.valid_metrics = MetricCollection([
            Accuracy(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
            F1Score(task="binary"),
            MatthewsCorrCoef(task="binary", num_classes=2),
            AUROC(task="binary"),
            AveragePrecision(task="binary")
        ], prefix='val_')

        self.train_metrics = self.valid_metrics.clone(prefix="train_")
        self.test_metrics = self.valid_metrics.clone(prefix="test_")

    def _single_step(self, batch):
        preds = self.forward(batch)
        preds = preds.view(-1)
        loss = F.binary_cross_entropy(preds, batch["label"].to(torch.float32))
        return batch["label"], preds, loss

    def training_step(self, batch, batch_idx):
        trues, preds, loss = self._single_step(batch)
        self.train_metrics.update(preds, trues)
        return loss

    def test_step(self, batch, batch_idx):
        trues, preds, test_loss = self._single_step(batch)
        self.test_metrics.update(preds, trues)
        self.log("test_loss", test_loss, batch_size=self.hparams.batch_size, sync_dist=self.hparams.sync_dist)

    def validation_step(self, batch, batch_idx):
        trues, preds, val_loss = self._single_step(batch)
        self.valid_metrics.update(preds, trues)
        self.log("val_loss", val_loss, batch_size=self.hparams.batch_size, sync_dist=self.hparams.sync_dist)

    def on_train_epoch_end(self):
        result = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(result, on_epoch=True, sync_dist=self.hparams.sync_dist)

    def on_test_epoch_end(self):
        result = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(result, on_epoch=True, sync_dist=self.hparams.sync_dist)

    def on_validation_epoch_end(self):
        result = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(result, on_epoch=True, sync_dist=self.hparams.sync_dist)

    def load_data(self, dataset, valid_size=0.1, indices=None):
        if indices is None:
            dataset_length = len(dataset)
            valid_length = int(valid_size * dataset_length)
            train_length = dataset_length - valid_length
            self.train_set, self.val_set = data.random_split(dataset, [train_length, valid_length])
            print('Data has been randomly divided into train/val sets with sizes {} and {}'.format(len(self.train_set),
                                                                                                   len(self.val_set)))
        else:
            train_indices, val_indices = indices
            self.train_set = Subset(dataset, train_indices)
            self.val_set = Subset(dataset, val_indices)
            print('Data has been divided into train/val sets with sizes {} and {} based on selected indices'.format(
                len(self.train_set), len(self.val_set)))

    def train_dataloader(self, train_set=None, num_workers=8):
        if train_set is not None:
            self.train_set = train_set
        return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=True)

    def test_dataloader(self, test_set=None, num_workers=8):
        if test_set is not None:
            self.test_set = test_set
        return DataLoader(dataset=self.test_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers)

    def val_dataloader(self, val_set=None, num_workers=8):
        if val_set is not None:
            self.val_set = val_set
        return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Args_model")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training. "
                                                                   "Cosine warmup will be applied.")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training/testing.")
        parser.add_argument("--encoder_features", type=int, default=1536,
                            # help="Number of features in the encoder "
                            #      "(Corresponds to the dimentionality of per-token embedding of ESM2 model.) "
                            #      "If not a 3B version of ESM2 is chosen, this parameter needs to be set accordingly."
                            help=argparse.SUPPRESS)
        return parent_parser


class AttentionModel(BaselineModel):
    def __init__(self, params):
        super(AttentionModel, self).__init__(params)

        # self.ankh_model, _ = ankh.load_large_model()
        # for param in self.ankh_model.parameters():
        #     param.requires_grad = False

        self.encoder_features = self.hparams.encoder_features

        self.dense_head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.encoder_features*2, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, batch):

        # emb1 = self.ankh_model(input_ids=batch['input_ids'][:, 0, :], attention_mask=batch['attention_mask'][:, 0, :]).last_hidden_state
        # emb2 = self.ankh_model(input_ids=batch['input_ids'][:, 1, :], attention_mask=batch['attention_mask'][:, 1, :]).last_hidden_state

        head1 = batch['emb_0'].mean(dim=1)
        head2 = batch['emb_1'].mean(dim=1)

        return self.dense_head(torch.cat([head1, head2], dim=1))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)
        lr_dict = {
            "scheduler": CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=200),
            "name": 'CosineWarmupScheduler',
        }
        return [optimizer], [lr_dict]


if __name__ == '__main__':
    from dataset import PairSequenceData
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

    dataset = PairSequenceData(actions_file="../SENSE-PPI/data/guo_yeast_data/protein.pairs.tsv",
                        sequences_file="../SENSE-PPI/data/guo_yeast_data/sequences.fasta",
                        max_len=400)
    
    print(len(dataset))
    
    parser = argparse.ArgumentParser()
    parser = BaselineModel.add_model_specific_args(parser)
    params = parser.parse_args()

    model = AttentionModel(params)

    model.load_data(dataset=dataset, valid_size=0.2)
    train_set = model.train_dataloader()
    val_set = model.val_dataloader()

    logger = pl.loggers.TensorBoardLogger("logs", name='AttentionModelBase')

    callbacks = [
        TQDMProgressBar(refresh_rate=50),
        ModelCheckpoint(filename='chkpt_loss_based_{epoch}-{val_loss:.3f}-{val_BinaryF1Score:.3f}', verbose=True,
                        monitor='val_loss', mode='min', save_top_k=1),
        EarlyStopping(monitor="val_loss", patience=10,
                                    verbose=False, mode="min")
    ]

    trainer = pl.Trainer(accelerator='gpu', num_nodes=1, max_epochs=100, 
                         logger=logger, callbacks=callbacks)

    trainer.fit(model, train_set, val_set)

    