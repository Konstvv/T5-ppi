import pandas as pd
from tqdm import tqdm

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


if __name__ == '__main__':
    # organisms_set_train, prots_set_train = get_sets('all_900_train.tsv')
    # organisms_set_test, prots_set_test = get_sets('all_900_test.tsv')

    # print('Organisms in train:', len(organisms_set_train))
    # print('Organisms in test:', len(organisms_set_test))

    # print('Proteins in train:', len(prots_set_train))
    # print('Proteins in test:', len(prots_set_test))

    # print('Common organisms:', len(organisms_set_train & organisms_set_test))
    # print('Common proteins:', len(prots_set_train & prots_set_test))
    from Bio import SeqIO
    with open('string12.0_combined_score_900.fasta') as f:
        with open('string12.0_combined_score_900_esm.fasta', 'w') as f2:
            for record in tqdm(SeqIO.parse(f, 'fasta')):
                record.seq = record.seq.replace('J', 'L')
                f2.write(f'>{record.id}\n')
                f2.write(f'{record.seq}\n')