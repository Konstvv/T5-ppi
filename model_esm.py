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
from dataset import PairSequenceData, SequencesDataset, PairSequenceDataIterable, PairSequenceDataPrecomputed
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
import wandb
from transformers import PreTrainedTokenizerFast

# from torch.nn import MultiheadAttention
from rope import RotaryPEMultiHeadAttention as MultiheadAttention

from pytorch_lightning.callbacks import Callback
import gc

from esm import FastaBatchedDataset, pretrained

from model import BaselineModel, CosineWarmupScheduler, MemoryCleanupCallback, CrossTransformerModule


class ESM2Module(torch.nn.Module):
    def __init__(self, 
                 params):
        super(ESM2Module, self).__init__()

        self.esm2_model, self.alphabet = pretrained.esm2_t12_35M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=params.max_len)

        if torch.cuda.is_available():
            self.model_device = 'cuda'
        else:
            self.model_device = 'cpu'

        self.esm2_model.to(self.model_device)

    def forward(self, x1, x2):

        x1, mask1 = self.get_embs(x1)
        x2, mask2 = self.get_embs(x2)

        return (x1, mask1), (x2, mask2)
        

    def get_embs(self, seqs):
        seqs = [('', seq) for seq in seqs]
        _, _, batch_tokens = self.batch_converter(seqs)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1).to(self.model_device)

        results = self.esm2_model(batch_tokens.to(self.model_device), repr_layers=[12])
        token_representations = results["representations"][12]

        # mask = (batch_tokens != self.alphabet.padding_idx).to(self.model_device)
        mask = torch.arange(token_representations.shape[1], device=self.model_device)[None, :] < batch_lens[:, None]

        return token_representations, mask
    
    def freeze_params(self):
        for param in self.esm2_model.parameters():
            param.requires_grad = False

    def unfreeze_params(self):
        for param in self.esm2_model.parameters():
            param.requires_grad = True
    

class PPITransformerModel(BaselineModel):
    def __init__(self, 
                 params, 
                 embed_dim : int = 1024,
                 hidden_dim: int = 2048,
                 num_cross_layers: int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 precomputed_embeds: bool = False):
        super(PPITransformerModel, self).__init__(params)
        

        self.embed_dim = embed_dim

        self.self_transformer_block = None
        if not precomputed_embeds:
            self.self_transformer_block = ESM2Module(params)
            self.self_transformer_block.freeze_params()

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

    def forward(self, batch, return_attention=False):
        
        if self.self_transformer_block is not None:
            x1 = batch['ids1']
            x2 = batch['ids2']
            (x1, mask1), (x2, mask2) = self.self_transformer_block(x1, x2)
        
        else:
            x1, mask1 = batch['ids1']
            x2, mask2 = batch['ids2']

        if return_attention:
            x, (attn1, attn2) = self.cross_transformer_block(x1, x2, (mask1, mask2), return_attention=return_attention)
        else:
            x = self.cross_transformer_block(x1, x2, (mask1, mask2))

        x = self.decoder(x)

        if return_attention:
            return x, (attn1, attn2)
        else:
            return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)

        lr_dict = {
            "scheduler": CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=int(1e7)),
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

    # sequences = SequencesDataset(sequences_path="dev.fasta")

    # dataset = PairSequenceData(pairs_path="dev_train.tsv",
    #                             sequences_dataset=sequences, for_esm=True)

    # dataset_test = PairSequenceData(pairs_path="dev_test.tsv",
    #                                 sequences_dataset=sequences, for_esm=True)

    # sequences = SequencesDataset(sequences_path="string12.0_combined_score_900_esm.fasta")

    # dataset = PairSequenceData(pairs_path="all_900_train_shuffled.tsv",
    #                             sequences_dataset=sequences, for_esm=True)

    # dataset_test = PairSequenceData(pairs_path="all_900_test.tsv",
    #                                 sequences_dataset=sequences, for_esm=True)
    
    dataset = PairSequenceDataPrecomputed(pairs_path="all_900_train_shuffled.tsv", emb_dir="esm2_embs_3B", nrows=200000000)
    dataset_test = PairSequenceDataPrecomputed(pairs_path="all_900_test.tsv", emb_dir="esm2_embs_3B")


    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = 1000 # dataset.max_len

    #define sync_dist parameter for distributed training
    if 'ddp' in str(params.strategy):
        params.sync_dist = True
        print('Distributed syncronisation is enabled')

    model = PPITransformerModel(params, 
                           embed_dim=480,
                            hidden_dim=2048,
                            num_cross_layers=6,
                            num_heads=8,
                            dropout=0.1,
                            precomputed_embeds=True)

    ckpt = torch.load("ppi-transformer/tmg60lyz/checkpoints/chkpt_loss_based_epoch=0-val_loss=0.078-val_BinaryF1Score=0.843.ckpt")
    for key in list(ckpt['state_dict'].keys()):
        if key.startswith('embedding') or key.startswith('self_transformer_block'):
            del ckpt['state_dict'][key]
    model.load_state_dict(ckpt['state_dict'])

    # model.load_data(dataset=dataset, valid_size=0.01)
    shuffle_train = False if isinstance(dataset, PairSequenceDataIterable) else True
    train_set = model.train_dataloader(dataset, collate_fn=dataset.collate_fn, num_workers=params.num_workers, shuffle=shuffle_train)
    val_set = model.val_dataloader(dataset_test, collate_fn=dataset.collate_fn, num_workers=params.num_workers, shuffle=False)

    logger = pl.loggers.WandbLogger(project='ppi-transformer', name='ESM2precomp')

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
                         val_check_interval=50000,
                         limit_val_batches=0.5)

    trainer.fit(model, train_set, val_set)

    wandb.finish()
