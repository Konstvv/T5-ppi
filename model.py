import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, AveragePrecision
from torchmetrics.collections import MetricCollection
import torch.optim as optim
import numpy as np
import logging
from torch.optim import Optimizer
from dataset import PairSequenceData, SequencesDataset, PairSequenceDataIterable
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
import wandb
from transformers import PreTrainedTokenizerFast

# from torch.nn import MultiheadAttention
from rope import RotaryPEMultiHeadAttention as MultiheadAttention

from pytorch_lightning.callbacks import Callback
import gc

from esm import FastaBatchedDataset, pretrained

class MemoryCleanupCallback(Callback):
    def __init__(self):
        super().__init__()
        self.batch_idx = 0

    def on_train_batch_end(self, *args, **kwargs):
        #if batch index is multiple of 10000, do memory cleanup
        self.batch_idx += 1
        if self.batch_idx % 10000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            # logging.info('Memory cleanup done')
            # logging.info(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
            # logging.info(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2} MB")

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer: Optimizer,
                 warmup: int,
                 max_iters: int):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.num_batches = 0
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(batch=self.num_batches)
        # print('Current lr: ', [base_lr * lr_factor for base_lr in self.base_lrs])
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, batch: int):
        lr_factor = 0.5 * (1 + np.cos(np.pi * batch / self.max_num_iters))
        if batch <= self.warmup:
            lr_factor *= (batch + 1) * 1.0 / self.warmup
        return lr_factor

    def step(self, batch=None):
        if batch is None:
            self.num_batches += 1
        else:
            self.num_batches = batch

        super().step()


class BaselineModel(pl.LightningModule):
    def __init__(self, params):
        super(BaselineModel, self).__init__()

        self.save_hyperparameters(params)

        # Transfer to hyperparameters
        self.train_set = None
        self.val_set = None
        self.test_set = None

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
        labels = batch['labels']
        preds = self.forward(batch).view(-1)

        # Manually disable autocast for binary_cross_entropy
        with torch.amp.autocast("cuda", enabled=False):
            # Ensure preds and labels are in float32
            loss = F.binary_cross_entropy(preds.to(torch.float32), labels.to(torch.float32))
        
        return labels, preds, loss

    def training_step(self, batch, batch_idx):
        trues, preds, loss = self._single_step(batch)
        self.train_metrics.update(preds.detach(), trues.detach())

        # Reset metrics every N batches
        if (batch_idx + 1) % self.hparams.metric_reset_interval == 0:
            result = self.train_metrics.compute()
            self.train_metrics.reset()

            self.log_dict(result, on_step=True, sync_dist=self.hparams.sync_dist)

            self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'],
                    on_step=True, on_epoch=False,
                    batch_size=self.hparams.batch_size, sync_dist=self.hparams.sync_dist)

            self.log("train_loss", loss, on_step=True, sync_dist=self.hparams.sync_dist)

        return loss

    def test_step(self, batch, batch_idx):
        trues, preds, test_loss = self._single_step(batch)
        self.test_metrics.update(preds.detach(), trues.detach())
        self.log("test_loss", test_loss, batch_size=self.hparams.batch_size, sync_dist=self.hparams.sync_dist)

    def validation_step(self, batch, batch_idx):
        trues, preds, val_loss = self._single_step(batch)
        self.valid_metrics.update(preds.detach(), trues.detach())
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

    def train_dataloader(self, train_set=None, num_workers=8, collate_fn=None, shuffle=True):
        if train_set is not None:
            self.train_set = train_set
        if isinstance(self.train_set, PairSequenceDataIterable):
            return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          prefetch_factor=2)
        return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          pin_memory=True,
                          prefetch_factor=2)

    def test_dataloader(self, test_set=None, num_workers=8, collate_fn=None, shuffle=True):
        if test_set is not None:
            self.test_set = test_set
        if isinstance(self.test_set, PairSequenceDataIterable):
            return DataLoader(dataset=self.test_set,
                          batch_size=self.hparams.batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=True,
                          prefetch_factor=2)
        return DataLoader(dataset=self.test_set,
                          batch_size=self.hparams.batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          pin_memory=True,
                          prefetch_factor=2)

    def val_dataloader(self, val_set=None, num_workers=8, collate_fn=None, shuffle=True):
        if val_set is not None:
            self.val_set = val_set
        if isinstance(self.val_set, PairSequenceDataIterable):
            return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=True,
                          prefetch_factor=2)
        return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          pin_memory=True,
                          prefetch_factor=2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Args_model")
        parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training. "
                                                                   "Cosine warmup will be applied.")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/testing.")
        parser.add_argument("--max_len", type=int, default=800, help="Max sequence length.")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
        parser.add_argument("--sync_dist", type=bool, default=False, help="Synchronize distributed training.")
        parser.add_argument("--metric_reset_interval", type=int, default=10000, help="Interval to reset metrics")
        return parent_parser


class PositionwiseFeedForward(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 dropout: float):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, input_dim)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))


class SelfTransformerLayer(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.2,
                 precision: int = 32):
        super().__init__()
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.norm3 = torch.nn.LayerNorm(input_dim)
        self.attn = MultiheadAttention(input_dim, num_heads, dropout, precision=precision)
        self.ffn = PositionwiseFeedForward(input_dim, hidden_dim, dropout)

    def forward(self, x, mask=None):
        x_add = torch.add(x, self.attn(x, x, x, mask=mask))
        x_ffn = self.ffn(self.norm2(x_add))
        return self.norm3(torch.add(x_ffn, x_add))


class CrossTransformerLayer(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.2,
                 precision: int = 32):
        super().__init__()
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.norm3 = torch.nn.LayerNorm(input_dim)
        self.cross_attention = MultiheadAttention(input_dim, num_heads, dropout, precision=precision)
        self.ffn = PositionwiseFeedForward(input_dim, hidden_dim, dropout)
        
        self.attn1 = None
        self.attn2 = None

    def forward(self, x1, x2, masks=None):
        x1_add = self.cross_attention(x1, x2, x2, mask=masks)
        self.attn1 = self.cross_attention.attn.detach()

        x2_add = self.cross_attention(x2, x1, x1, mask=(masks[1], masks[0]))
        self.attn2 = self.cross_attention.attn.detach()

        x1_add = torch.add(x1, x1_add)
        x2_add = torch.add(x2, x2_add)

        x1_ffn = self.ffn(self.norm2(x1_add))
        x2_ffn = self.ffn(self.norm2(x2_add))

        x1_out = self.norm3(torch.add(x1_ffn, x1_add))
        x2_out = self.norm3(torch.add(x2_ffn, x2_add))

        return x1_out, x2_out


class SelfTransformerModule(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.2, 
                 num_layers: int = 1,
                 precision: int = 32):
        
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dim)

        self.self_transformer_layers = torch.nn.ModuleList(
            [SelfTransformerLayer(input_dim, num_heads, hidden_dim, dropout, precision=precision) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        x = self.norm(x)
        if self.self_transformer_layers:
            for layer in self.self_transformer_layers:
                x = layer(x, mask)

        return x


class CrossTransformerModule(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.2, 
                 num_layers: int = 1,
                 pooling: str = 'max',
                 precision: int = 32):
        
        super().__init__()

        self.norm = torch.nn.LayerNorm(input_dim)

        self.cross_transformer_layers = torch.nn.ModuleList(
            [CrossTransformerLayer(input_dim, num_heads, hidden_dim, dropout, precision=precision) for _ in range(num_layers)]
        )

        if pooling == 'mean':
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif pooling == 'max':
            self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('Supported pooling methods are "mean" and "max"')

    def forward(self, x1, x2, masks=None, return_attention=False):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        if self.cross_transformer_layers:
            for layer in self.cross_transformer_layers:
                x1, x2 = layer(x1, x2, masks)
                if return_attention:
                    attn1 = layer.attn1
                    attn2 = layer.attn2

        x = torch.cat([x1, x2], dim=1)

        x_out = self.pooling(x.permute(0, 2, 1)).squeeze()

        if return_attention:
            return x_out, (attn1, attn2)
        return x_out
    

class PPITransformerModel(BaselineModel):
    def __init__(self, 
                 params, 
                 ntoken: int = 32, 
                 embed_dim : int = 1024,
                 hidden_dim: int = 2048,
                 num_siamese_layers: int = 12,
                 num_cross_layers: int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super(PPITransformerModel, self).__init__(params)
        

        self.embed_dim = embed_dim

        self.embedding = torch.nn.Embedding(ntoken, self.embed_dim)

        self.self_transformer_block = SelfTransformerModule(input_dim=self.embed_dim,
                                                            num_heads=num_heads,
                                                            hidden_dim=hidden_dim,
                                                            num_layers=num_siamese_layers,
                                                            dropout=dropout,
                                                            precision=self.hparams.precision)

        self.cross_transformer_block = CrossTransformerModule(input_dim=self.embed_dim, 
                                                              num_heads=num_heads, 
                                                              hidden_dim=hidden_dim, 
                                                              num_layers=num_cross_layers, 
                                                              dropout=dropout,
                                                              pooling='mean',
                                                                precision=self.hparams.precision)

        self.decoder = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.embed_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, batch):
        x1, mask1 = batch['ids1']
        x2, mask2 = batch['ids2']

        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        x1 = self.self_transformer_block(x1, mask1)
        x2 = self.self_transformer_block(x2, mask2)

        x = self.cross_transformer_block(x1, x2, (mask1, mask2))

        x = self.decoder(x)

        return x
    
    def freeze_self_transformer(self):
        for param in self.self_transformer_block.parameters():
            param.requires_grad = False

    def unfreeze_self_transformer(self):
        for param in self.self_transformer_block.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01) #TODO: change the wd bask to 0.01

        lr_dict = {
            "scheduler": CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=int(1e7)),
            "name": 'CosineWarmupScheduler',
            "interval": 'step',
        }
        return [optimizer], [lr_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = BaselineModel.add_model_specific_args(parent_parser)
        return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # tokenizer = PPITokenizer()
    # _, tokenizer = ankh.load_base_model()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer")

    # sequences = SequencesDataset(sequences_path="dev.fasta")

    # dataset = PairSequenceData(pairs_path="dev_train.tsv",
    #                             sequences_dataset=sequences, tokenizer=tokenizer)

    # dataset_test = PairSequenceData(pairs_path="dev_test.tsv",
    #                                 sequences_dataset=sequences, tokenizer=tokenizer)

    sequences = SequencesDataset(sequences_path="string12.0_combined_score_900.fasta")

    dataset = PairSequenceData(pairs_path="all_900_train_shuffled.tsv",
                                sequences_dataset=sequences, tokenizer=tokenizer)#, chunk_size=100000)

    dataset_test = PairSequenceData(pairs_path="all_900_test.tsv",
                                    sequences_dataset=sequences, tokenizer=tokenizer)


    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = dataset.max_len

    #define sync_dist parameter for distributed training
    if 'ddp' in str(params.strategy):
        params.sync_dist = True
        print('Distributed syncronisation is enabled')

    model = PPITransformerModel(params, 
                           ntoken=len(dataset.tokenizer), 
                           embed_dim=512, #64/ #512
                            hidden_dim=2048, #512/ #2048
                            num_siamese_layers=12, #6
                            num_cross_layers=6, #3
                            num_heads=8, #8
                            dropout=0.1)

    # ckpt = torch.load("ppi-transformer/6qfgdx0p/checkpoints/chkpt_loss_based_epoch=0-val_loss=0.107-val_BinaryF1Score=0.781.ckpt")
    # model.load_state_dict(ckpt['state_dict'])

    # model.load_data(dataset=dataset, valid_size=0.01)
    shuffle_train = False if isinstance(dataset, PairSequenceDataIterable) else True
    train_set = model.train_dataloader(dataset, collate_fn=dataset.collate_fn, num_workers=params.num_workers, shuffle=shuffle_train)
    val_set = model.val_dataloader(dataset_test, collate_fn=dataset.collate_fn, num_workers=params.num_workers, shuffle=False)

    # logger = pl.loggers.TensorBoardLogger("logs", name='AttentionModelBase')
    logger = pl.loggers.WandbLogger(project='ppi-transformer', name='ESM2')

    callbacks = [
        TQDMProgressBar(refresh_rate=250),
        ModelCheckpoint(filename='chkpt_loss_based_{epoch}-{val_loss:.3f}-{val_BinaryF1Score:.3f}', verbose=True,
                        monitor='val_loss', mode='min', save_top_k=1),
        EarlyStopping(monitor="val_loss", patience=5,
                      verbose=False, mode="min"),
        MemoryCleanupCallback()
    ]

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto', 
                         num_nodes=params.num_nodes,
                         strategy=params.strategy,
                         devices=params.devices,
                         accumulate_grad_batches=params.accumulate_grad_batches,
                         max_epochs=100,
                         logger=logger, 
                         callbacks=callbacks,
                         precision=params.precision,
                         track_grad_norm=2,
                        val_check_interval=100000)

    trainer.fit(model, train_set, val_set)

    wandb.finish()
