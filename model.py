import argparse
from typing import List

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
from transformers import PreTrainedTokenizer, AutoTokenizer
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers, processors, models


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
        self.hparams.sync_dist = False
        # mandatory check that self.hparams has devices attribute and if it is not NoneType then is is > 1
        if hasattr(self.hparams, 'devices'):
            if self.hparams.devices is not None and int(self.hparams.devices) > 1:
                print('Using distributed training with {} gpus'.format(self.hparams.devices))
                self.hparams.sync_dist = True

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

    def train_dataloader(self, train_set=None, num_workers=16):
        if train_set is not None:
            self.train_set = train_set
        return DataLoader(dataset=self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=True)

    def test_dataloader(self, test_set=None, num_workers=16):
        if test_set is not None:
            self.test_set = test_set
        return DataLoader(dataset=self.test_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers)

    def val_dataloader(self, val_set=None, num_workers=16):
        if val_set is not None:
            self.val_set = val_set
        return DataLoader(dataset=self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Args_model")
        parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training. "
                                                                   "Cosine warmup will be applied.")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/testing.")
        parser.add_argument("--max_len", type=int, default=800, help="Max sequence length.")
        # parser.add_argument("--encoder_features", type=int, default=768,
        #                     # help="Number of features in the encoder "
        #                     #      "(Corresponds to the dimentionality of per-token embedding of ESM2 model.) "
        #                     #      "If not a 3B version of ESM2 is chosen, this parameter needs to be set accordingly."
        #                     help=argparse.SUPPRESS)
        return parent_parser


class CrossAttentionModule(pl.LightningModule):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention1 = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attention2 = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention1 = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention2 = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, h1, h2):
        attn1, _ = self.self_attention1(h1, h1, h1)
        attn2, _ = self.self_attention2(h2, h2, h2)
        x1 = h1 + attn1
        x2 = h2 + attn2

        cross_attn1, _ = self.cross_attention1(x1, h2, h2)
        cross_attn2, _ = self.cross_attention2(x2, h1, h1)
        cross1 = x1 + cross_attn1
        cross2 = x2 + cross_attn2
        return cross1, cross2


class PositionwiseFeedForward(pl.LightningModule):
    def __init__(self, embed_dim, hidden_dim, dropout_p):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SelfTransformerBlock(pl.LightningModule):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_p):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim, hidden_dim, dropout_p)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossTransformerBlock(pl.LightningModule):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_p, merge=False):
        super().__init__()
        self.merge = merge
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attn = CrossAttentionModule(embed_dim, num_heads)
        self.ffn1 = PositionwiseFeedForward(embed_dim, hidden_dim, dropout_p)
        if not self.merge:
            self.norm12 = torch.nn.LayerNorm(embed_dim)
            self.norm22 = torch.nn.LayerNorm(embed_dim)
            self.ffn2 = PositionwiseFeedForward(embed_dim, hidden_dim, dropout_p)

    def forward(self, x1, x2, h12_acc=None):
        h1, h2 = self.attn(x1, x2)

        if h12_acc is not None:
            h12_acc += h1 + h2
        else:
            h12_acc = h1 + h2

        if self.merge:
            x = self.norm1(h12_acc)
            x = self.norm2(x + self.ffn1(x))
            return x
        else:
            x1 = self.norm1(x1 + h1)
            x2 = self.norm2(x2 + h2)
            x1 = self.norm12(x1 + self.ffn1(x1))
            x2 = self.norm22(x2 + self.ffn2(x2))
            return x1, x2, h12_acc


class PositionalEncoding(pl.LightningModule):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class AttentionModel(BaselineModel):
    def __init__(self, params):
        super(AttentionModel, self).__init__(params)

        self.tokenizer = create_tokenizer()#Tokenizer.from_file("tokenizer.json")

        self.positional_encoding = PositionalEncoding(params.max_len, max_len=params.max_len)
        self.transformer_blocks = torch.nn.Sequential(*[SelfTransformerBlock(params.max_len, 8, 2048, 0.1) for _ in range(12)])
        self.cross_transformer_block = CrossTransformerBlock(params.max_len, 8, 2048, 0.1, merge=True)


        self.dense_head = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.2),
            # torch.nn.Linear(params.max_len, 32),
            # torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(params.max_len, 1),
            torch.nn.Sigmoid()
        )

    def batch_encode(self, seqs: List[str], max_length=None):
        ids = [encoding.ids for encoding in self.tokenizer.encode_batch(seqs)]

        if max_length is None:
            max_length = max(len(id) for id in ids)

        pad_id = self.tokenizer.token_to_id("[PAD]")

        return torch.cat([F.pad(torch.tensor(id), (0, max_length - len(id)), value=pad_id).unsqueeze(0) for id in ids])

    def forward(self, batch):

        tok1 = self.batch_encode(batch["seq1"], max_length=self.hparams.max_len).to(params.accelerator)
        tok2 = self.batch_encode(batch["seq2"], max_length=self.hparams.max_len).to(params.accelerator)

        x1 = self.positional_encoding(self.transformer_blocks(tok1.unsqueeze(0).to(torch.float32)))
        x2 = self.positional_encoding(self.transformer_blocks(tok2.unsqueeze(0).to(torch.float32)))

        x = self.cross_transformer_block(x1, x2)
        x = self.dense_head(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        lr_dict = {
            "scheduler": CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=200),
            "name": 'CosineWarmupScheduler',
        }
        return [optimizer], [lr_dict]

def create_tokenizer():
        vocab = [text.strip() for text in open("vocab.txt").readlines()]

        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(vocab)
        tokenizer.unk_token = "[UNK]"
        tokenizer.normalizer = normalizers.Sequence([normalizers.Strip(),
                                               normalizers.Replace("\n", "")])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel()])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save("tokenizer.json")
        return tokenizer


if __name__ == '__main__':
    from dataset import PairSequenceData
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
    from torchsummary import summary
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    max_len = 800

    dataset = PairSequenceData(actions_file="../SENSE-PPI/data/dscript_data/human_train.tsv",
                               sequences_file="../SENSE-PPI/data/dscript_data/human.fasta",
                               max_len=max_len)

    print(len(dataset))

    parser = argparse.ArgumentParser()
    parser = BaselineModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = max_len
    # params.devices = 1
    params.accelerator = "cuda"

    model = AttentionModel(params)

    # print(summary(model, (max_len)))

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

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator=params.accelerator, num_nodes=params.devices,
                         max_epochs=100,
                         logger=logger, callbacks=callbacks)

    trainer.fit(model, train_set, val_set)


#version 4,8 - 1e-5
#version 7 - 1e-6
#version 9 - 1e-4