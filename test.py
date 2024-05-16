import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import AttentionModel

from dataset import PairSequenceData, SequencesDataset
import os


def test_model(checkpoint_folder, dataset):
    parser = argparse.ArgumentParser()
    parser = AttentionModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = 1002

    model = AttentionModel(params, ntoken=len(dataset.tokenizer), embed_dim=256)

    checkpoint_folder = os.path.join(checkpoint_folder, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator=params.accelerator, devices=params.devices, num_nodes=params.num_nodes)

    pred_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=dataset.collate_fn)
    trainer.test(model, pred_loader)

def string_select_data(filter_score=500, min_len=0, max_len=1000, output_name='string12.0_experimental_score_500'):
    import dask.dataframe as dd
    from dask.distributed import Client
    import dask
    from Bio import SeqIO
    from tqdm import tqdm
    all_prots = [record for record in tqdm(SeqIO.parse('protein.sequences.v12.0.fa', 'fasta'))]
    print('Total of {} proteins in fasta'.format(len(all_prots)))
    undesired_prots = [record.id for record in tqdm(all_prots) if len(record.seq) < min_len or len(record.seq) > max_len]
    print('Total of {} undesired proteins with lengths <{} and >{}:'.format(len(undesired_prots), min_len, max_len))
    print('Dask:', dask.__version__)
    client = Client(n_workers=74, threads_per_worker=1, memory_limit='32GB')
    print('Client:', client)
    data = dd.read_csv('/home/volzhenin/SENSE-PPI/data/protein.physical.links.full.v12.0.txt', sep=' ')
    print('Data:', data)

    data = data[data['experiments'] > filter_score]
    to_drop = data['protein1'].isin(undesired_prots) | data['protein2'].isin(undesired_prots)
    data = data[~to_drop]
    data = data.compute()
    data.to_csv('{}.tsv'.format(output_name), index=False, sep=' ')

    left_out_prots = set(data['protein1'].unique()) | set(data['protein2'].unique())
    print('Total of {} proteins written to tsv with a total pair count of {}'.format(len(left_out_prots), len(data)))
    max_written_len = 0
    count_written = 0
    with open('{}.fasta'.format(output_name), 'w') as f:
        for record in tqdm(all_prots):
            if record.id in left_out_prots:
                SeqIO.write(record, f, 'fasta')
                count_written += 1
                max_written_len = max(max_written_len, len(record.seq))
    print('Total of {} proteins written to fasta with max length of {}'.format(count_written, max_written_len))

def create_negatives_and_split(file):
    import pandas as pd
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False, nb_workers=32)
    data = pd.read_csv(file+'.tsv', sep=' ', usecols=['protein1', 'protein2', 'combined_score'])
    data['combined_score'] = 1

    data_test = data.sample(frac=0.1)
    data_train = data.drop(data_test.index)

    prots_from_fasta = set(data['protein1'].unique()) | set(data['protein2'].unique())
    print(len(prots_from_fasta))

    data_test_negatives = np.random.choice(list(prots_from_fasta), (len(data_test)*11, 2), replace=True)
    data_test_negatives = pd.DataFrame(data_test_negatives, columns=['protein1', 'protein2'])
    data_test_negatives['combined_score'] = 0

    data_test_negatives = data_test_negatives[data_test_negatives['protein1'] != data_test_negatives['protein2']]
    data_test_negatives = data_test_negatives.parallel_apply(lambda row: row if row['protein1'] < row['protein2'] else [row['protein2'], row['protein1'], row['combined_score']], axis=1, result_type='expand')
    data_test_negatives = data_test_negatives.drop_duplicates()

    data_test = pd.concat([data_test, data_test_negatives], ignore_index=True)
    data_test = data_test.sample(frac=1)
    data_test.to_csv(file+'_test.tsv', sep='\t', index=False, header=False)

    data_train_negatives = np.random.choice(list(prots_from_fasta), (len(data_train)*11, 2), replace=True)
    data_train_negatives = pd.DataFrame(data_train_negatives, columns=['protein1', 'protein2'])
    data_train_negatives['combined_score'] = 0

    data_train_negatives = data_train_negatives[data_train_negatives['protein1'] != data_train_negatives['protein2']]
    data_train_negatives = data_train_negatives.parallel_apply(lambda row: row if row['protein1'] < row['protein2'] else [row['protein2'], row['protein1'], row['combined_score']], axis=1, result_type='expand')
    data_train_negatives = data_train_negatives.drop_duplicates()

    data_train = pd.concat([data_train, data_train_negatives], ignore_index=True)
    data_train = data_train.sample(frac=1)
    data_train.to_csv(file+'_train.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':

    # string_select_data()

    # file = 'string12.0_experimental_score_500'
    # # file = 'string12.0_combined_score_700'
    # create_negatives_and_split(file)

    sequences = SequencesDataset(sequences_path="/home/volzhenin/T5-ppi/string12.0_combined_score_700_experimantal_less_500_sample.fasta")

    dataset = PairSequenceData(pairs_path="/home/volzhenin/T5-ppi/string12.0_combined_score_700_experimantal_less_500_sample.tsv",
                                sequences_dataset=sequences, max_len=1000)
    
    test_model(checkpoint_folder='logs/AttentionModelBase/version_10', dataset=dataset)