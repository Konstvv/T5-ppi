import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Bio import SeqIO

def get_sets(name):
    if 'train' in name:
        data = pd.read_csv(name, sep='\t', header=None, names=['prot1', 'prot2', 'label'], nrows=100000000, chunksize=100000)
    else:
        data = pd.read_csv(name, sep='\t', header=None, names=['prot1', 'prot2', 'label'], chunksize=100000)

    organisms_set = set()
    prots_set = set()
    for chunk in tqdm(data):
        chunk['org1'] = chunk['prot1'].apply(lambda x: x.split('.')[0])
        chunk['org2'] = chunk['prot2'].apply(lambda x: x.split('.')[0])

        organisms_set_chunk = set(chunk['org1'].unique()) | set(chunk['org2'].unique())
        prots_set_chunk = set(chunk['prot1'].unique()) | set(chunk['prot2'].unique())

        organisms_set |= organisms_set_chunk
        prots_set |= prots_set_chunk

    return organisms_set, prots_set

def get_organisms():
    organisms_set_train, prots_set_train = get_sets('all_900_train.tsv')
    organisms_set_test, prots_set_test = get_sets('all_900_test.tsv')

    print('Organisms in train:', len(organisms_set_train))
    print('Organisms in test:', len(organisms_set_test))

    print('Proteins in train:', len(prots_set_train))
    print('Proteins in test:', len(prots_set_test))

    print('Common organisms:', len(organisms_set_train & organisms_set_test))
    print('Common proteins:', len(prots_set_train & prots_set_test))

def get_pos_neg_dist():
    print('Reading data...')
    # data = pd.read_csv('all_900_train_shuffled.tsv', sep='\t', header=None, names=['prot1', 'prot2', 'label'])
    data = pd.read_csv('/home/volzhenin/SENSE-PPI/data/string11.5_neighbor_exclusion/protein.pairs_9606.tsv', sep='\t', header=None, names=['prot1', 'prot2', 'label'])
    
    data_positives = data[data['label'] == 1]
    data_negatives = data[data['label'] == 0]

    data_positives_count1 = data_positives['prot1'].value_counts()
    data_positives_count2 = data_positives['prot2'].value_counts()
    data_positives_count = data_positives_count1.add(data_positives_count2, fill_value=0)

    data_negatives_count1 = data_negatives['prot1'].value_counts()
    data_negatives_count2 = data_negatives['prot2'].value_counts()
    data_negatives_count = data_negatives_count1.add(data_negatives_count2, fill_value=0)

    
    # # with open('prot_dict_positives.pkl', 'wb') as f:
    # #     pickle.dump(data_positives_count, f)

    # # with open('prot_dict_negatives.pkl', 'wb') as f:
    # #     pickle.dump(data_negatives_count, f)

    # with open('prot_dict_positives.pkl', 'rb') as f:
    #     data_positives_count = pickle.load(f)

    # with open('prot_dict_negatives.pkl', 'rb') as f:
    #     data_negatives_count = pickle.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].hist(data_positives_count.to_list(), bins=500)
    ax[0].set_title('Positives', fontsize=20)
    ax[0].set_xlim(0, 100)
    ax[1].hist(data_negatives_count.to_list(), bins=50)
    ax[1].set_title('Negatives', fontsize=20)
    plt.savefig('hist_positives_negatives_senseppi.pdf', bbox_inches='tight')

    print(np.mean(data_positives_count.to_list()))
    print(np.mean(data_negatives_count.to_list()))

    print(np.median(data_positives_count.to_list()))
    print(np.median(data_negatives_count.to_list()))

def fasta_to_esm_standard():
    with open('string12.0_combined_score_900.fasta') as f:
        with open('string12.0_combined_score_900_esm.fasta', 'w') as f2:
            for record in tqdm(SeqIO.parse(f, 'fasta')):
                record.seq = record.seq.replace('J', 'L')
                f2.write(f'>{record.id}\n')
                f2.write(f'{record.seq}\n')

if __name__ == '__main__':
    # # split into 16 files
    # from Bio import SeqIO
    # records = []
    # with open('string12.0_combined_score_900_esm.fasta') as f:
    #     for record in tqdm(SeqIO.parse(f, 'fasta')):
    #         records.append(record)
    
    # for i in tqdm(range(16)):
    #     with open(f'string12.0_combined_score_900_esm_splits/string12.0_combined_score_900_esm_{i}.fasta', 'w') as f:
    #         records_to_write = records[i::16]
    #         SeqIO.write(records_to_write, f, 'fasta')
            

    #check the size of esm2_embs_3B/23.BEL05_03175.pt and esm2_embs_35M/23.BEL05_03175.pt

    import os

    