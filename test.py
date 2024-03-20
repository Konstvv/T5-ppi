import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import AttentionModel

if __name__ == '__main__':
    from dataset import PairSequenceData
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    max_len = 800

    dataset = PairSequenceData(actions_file="../SENSE-PPI/data/dscript_data/human_train.tsv",
                               sequences_file="../SENSE-PPI/data/dscript_data/human.fasta",
                               max_len=max_len-2)

    dataset_test = PairSequenceData(actions_file="../SENSE-PPI/data/dscript_data/human_test.tsv",
                                    sequences_file="../SENSE-PPI/data/dscript_data/human.fasta",
                                    max_len=max_len-2)

    parser = argparse.ArgumentParser()
    parser = AttentionModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = max_len
    # params.devices = 1
    params.accelerator = "gpu"

    model = AttentionModel(params, ntoken=len(dataset.tokenizer), embed_dim=256)

    checkpoint_folder = "logs/AttentionModelBase/version_4/checkpoints"
    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator=params.accelerator, num_nodes=params.devices)

    pred_loader = DataLoader(dataset=dataset_test, batch_size=32, num_workers=8, shuffle=False)
    trainer.test(model, pred_loader)

# def string_select_data():
#     import dask.dataframe as dd
#     from dask.distributed import Client
#     import dask
#     print('Dask:', dask.__version__)
#     client = Client(n_workers=64, threads_per_worker=1, memory_limit='32GB')
#     print('Client:', client)
#     data = dd.read_csv('../SENSE-PPI/protein.physical.links.full.v12.0.txt', sep=' ')
#     print('Data:', data)
#     data = data[data['combined_score'] > 700]

    
#     data = data.compute()

#     data.to_csv('string12.0_combined_score_700.tsv', index=False, sep=' ')

# if __name__ == '__main__':
#     from Bio import SeqIO
#     import pandas as pd
#     import os

#     data = pd.read_csv('interactions_intermediate.tmp', sep='\t')

#     data = data[['protein1', 'protein2']]

#     print('positive pairs: ', len(data))

#     proteins = set(data['protein1'].unique()).union(set(data['protein2'].unique()))

#     # randomly create negative pairs by sampling protein1 and protein2 from proteins, combined_score = 0
#     #use vectorized operations, the total length of the negative pairs is 11 times the length of the positive pairs
#     data_neg = pd.DataFrame({'protein1': pd.Series(list(proteins)).sample(n=11*len(data), replace=True).values,
#                             'protein2': pd.Series(list(proteins)).sample(n=11*len(data), replace=True).values})
#     #remove the pairs (p1, p2) that are in the positive pairs
#     data_neg = data_neg[~data_neg.isin(data)].dropna()
#     data_neg = data_neg.iloc[:10*len(data)]

#     print('negative pairs: ', len(data_neg))

#     data['label'] = 1
#     data_neg['label'] = 0

#     data = pd.concat([data, data_neg])
#     data.to_csv('protein.pairs_custom.tsv', index=False, sep='\t')

#     # string_select_data()

#     # with open('../SENSE-PPI/protein.sequences.v12.0.fasta', 'r') as f:
#     #     records = list(SeqIO.parse(f, 'fasta'))
#     #     record_ids = [r.id for r in records]
    
#     # print('ids retrieved:', len(record_ids))

#     # data = pd.read_csv('string12.0_combined_score_700.tsv', sep=' ')

#     # data = data[data['protein1'].isin(record_ids) & data['protein2'].isin(record_ids)]

#     # data.to_csv('string12.0_combined_score_700.tsv', index=False, sep=' ')

#     # print('tsv saved')

#     # data_prots = set(data['protein1'].unique()).union(set(data['protein2'].unique()))

#     # print('proteins in data:', len(data_prots))
 
#     # with open('../SENSE-PPI/protein.sequences.v12.0.fasta', 'r') as f:
#     #     with open('string12.0_combined_score_700.fasta', 'w') as f_out:
#     #         for record in SeqIO.parse(f, 'fasta'):
#     #             if record.id in data_prots:
#     #                 SeqIO.write(record, f_out, 'fasta')