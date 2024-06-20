import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import PPITransformerModel
import ankh

from dataset import PairSequenceData, SequencesDataset
import os


def test_model(checkpoint_folder, dataset):
    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = 1002

    model = PPITransformerModel(params, 
                        ntoken=len(dataset.tokenizer), 
                        embed_dim=512, #512
                        hidden_dim=2048, #2048
                        num_siamese_layers=6, #6
                        num_cross_layers=3, #3
                        num_heads=8, #8
                        dropout=0.1)

    checkpoint_folder = os.path.join(checkpoint_folder, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    print('Loading model from checkpoint:', checkpoint_path)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto',
                          devices=params.devices, 
                          num_nodes=params.num_nodes)

    pred_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=dataset.collate_fn)
    trainer.test(model, pred_loader)

if __name__ == '__main__':

    # file='string12.0_combined_score_900'
    # string_select_data(filter_score=900, min_len=0, max_len=1000, output_name='string12.0_combined_score_900')
    # create_negatives_and_split(file, pos_neg_ratio=1, test_frac=0.01)
    
    _, tokenizer = ankh.load_base_model()
    sequences = SequencesDataset(sequences_path="aphid_small.fasta")

    dataset = PairSequenceData(pairs_path="aphid_small.tsv",
                                sequences_dataset=sequences, tokenizer=tokenizer, max_len=800)
    
    test_model(checkpoint_folder='logs/AttentionModelBase/version_171', dataset=dataset)