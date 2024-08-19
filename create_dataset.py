import pandas as pd
import multiprocessing
# import modin.pandas as pd

import os
from Bio import SeqIO
from tqdm import tqdm
from Bio import SeqIO
import logging
import argparse
import multiprocessing
import tempfile
import requests
import json
import subprocess
from urllib.error import HTTPError
import wget
import gzip
import shutil
import random
import pickle

DOWNLOAD_LINK_STRING = "https://stringdb-downloads.org/download/"

def get_string_url():
    # Get stable api and current STRING version
    request_url = "/".join(["https://string-db.org/api", "json", "version"])
    response = requests.post(request_url)
    version = json.loads(response.text)[0]['string_version']
    stable_address = json.loads(response.text)[0]['stable_address']
    return "/".join([stable_address, "api"]), version


# A class containing methods for preprocessing the full STRING data
class STRINGDatasetCreation:
    def __init__(self, params):
        self.params = params
        self.interactions_file = params.interactions
        self.sequences_file = params.sequences
        self.min_length = params.min_length
        self.max_length = params.max_length
        self.species = 'custom' if params.species is None else params.species
        self.max_positive_pairs = params.max_positive_pairs
        self.combined_score = params.combined_score
        self.experimental_score = params.experimental_score

        self.SAVE_SEQ_PATH = "sequences_{}.fasta".format(self.species)
        self.SAVE_PAIRS_PATH = "protein.pairs_{}.tsv".format(self.species)

        # self.preprocess_fasta_file()
        os.makedirs('pickles', exist_ok=True)
        if not os.path.exists('pickles/fasta_dict.pkl'):
            self.fasta_records = self.preprocess_fasta_file()
            with open('pickles/fasta_dict.pkl', 'wb') as f:
                pickle.dump(self.fasta_records, f)
        else:
            with open('pickles/fasta_dict.pkl', 'rb') as f:
                self.fasta_records = pickle.load(f)
            logging.info('Total number of proteins in the fasta file: {}'.format(len(self.fasta_records)))

        if not os.path.exists('pickles/clusters.pkl'):
            self.clusters = self._create_clusters()
            with open('pickles/clusters.pkl', 'wb') as f:
                pickle.dump(self.clusters, f)
        else:
            with open('pickles/clusters.pkl', 'rb') as f:
                self.clusters = pickle.load(f)
            logging.info('Proteins in clusters: {}'.format(len(self.clusters)))
            logging.info('Number of clusters: {}'.format(len(set(self.clusters.values()))))

        positive_interactions, proteins_dict = self.select_interactions_and_prots()
        # if not os.path.exists('pickles/int_prots.pkl'):
        #     positive_interactions, proteins_dict = self.select_interactions_and_prots()
        #     with open('pickles/int_prots.pkl', 'wb') as f:
        #         pickle.dump((positive_interactions, proteins_dict), f)
        # else:
        #     with open('pickles/int_prots.pkl', 'rb') as f:
        #         positive_interactions, proteins_dict = pickle.load(f)
        #     logging.info('Interactions loaded from pkl.')

        interactions = self.create_negatives(positive_interactions, proteins_dict)

        self.train_test_split(interactions, test_size=0.1)


    def _create_clusters(self):
        logging.info('Running mmseqs to clusterize proteins')
        logging.info('This might take a while if you decide to process the whole STRING database.')
        logging.info(
            'In order to install mmseqs (if not installed), please visit https://github.com/soedinglab/MMseqs2')
        with tempfile.TemporaryDirectory() as mmseqdbs:
            commands = "; ".join([f"mmseqs createdb {self.sequences_file} {mmseqdbs}/DB",
                                    f"mmseqs cluster {mmseqdbs}/DB {mmseqdbs}/clusterDB {mmseqdbs}/tmp --min-seq-id 0.4 --alignment-mode 3 --cov-mode 1 --threads {self.params.threads_per_worker}",
                                    f"mmseqs createtsv {mmseqdbs}/DB {mmseqdbs}/DB {mmseqdbs}/clusterDB {mmseqdbs}/clusters.tsv"
                                    ])
            ret = subprocess.run(commands, shell=True, capture_output=True)
            print(ret.stdout.decode())
            print(ret.stderr.decode())
            logging.info('Clusters data computed')
            clusters = pd.read_csv(f"{mmseqdbs}/clusters.tsv", sep='\t', header=None,
                            names=['cluster', 'protein']).set_index('protein')
        
        clusters = clusters.to_dict()['cluster']
    
        logging.info('Proteins in clusters: {}'.format(len(clusters)))
        logging.info('Number of clusters: {}'.format(len(set(clusters.values()))))
        return clusters

    def select_interactions_and_prots(self):

        logging.info('Removing interactions according to protein length and confidence scores.')

        infomsg = 'Extracting only entries with no homology' \
                    ', combined score >= {}'.format(self.combined_score)
        if self.experimental_score is not None:
            infomsg += ', experimental score >= {}'.format(self.experimental_score)
        logging.info(infomsg)

        # interactions = pd.read_csv(self.interactions_file, 
        #                            sep=' ', 
        #                            usecols=['protein1', 'protein2', 'combined_score', 'homology', 'experiments'])
        # if self.experimental_score is not None:
        #     interactions = interactions[interactions['experiments'] > self.experimental_score]
        # interactions = interactions[interactions['combined_score'] > self.combined_score]
        # interactions = interactions[interactions['homology'] == 0]
        # interactions = interactions[['protein1', 'protein2', 'combined_score']]
        # if self.species != 'custom':
        #     interactions = interactions[interactions['protein1'].str.startswith(self.species) & interactions['protein2'].str.startswith(self.species)]
        # interactions = interactions[interactions['protein1'].isin(self.fasta_records.keys()) & interactions['protein2'].isin(self.fasta_records.keys())]
        # interactions['combined_score'] = 1      


        # logging.info('Sorting positive protein pairs.')

        # interactions['protein1'], interactions['protein2'] = zip(*interactions.apply(
        #     lambda x: (x['protein1'], x['protein2']) if x['protein1'] < x['protein2'] else (
        #         x['protein2'], x['protein1']), axis=1))

        lines = []
        with open(self.interactions_file, 'r') as f:
            for line in tqdm(f):
                if line.startswith('protein1'):
                    continue
                line = line.strip().split()
                # protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score
                if int(line[-1]) < self.combined_score:
                    continue
                if self.experimental_score is not None and int(line[3]) < self.experimental_score:
                    continue
                if int(line[2]) > 0:
                    continue
                # if self.species != 'custom' and not (line[0].startswith(self.species) and line[1].startswith(self.species)):
                    # continue
                if line[0] not in self.fasta_records or line[1] not in self.fasta_records:
                    continue
                p1 = line[0] if line[0] < line[1] else line[1]
                p2 = line[1] if line[0] < line[1] else line[0]
                lines.append((p1, p2, 1))

        #dataframe from list of tuples
        interactions = pd.DataFrame(lines, columns=['protein1', 'protein2', 'combined_score'])

        interactions = interactions.drop_duplicates(subset=['protein1', 'protein2'], keep='first')

        interactions['clusters'] = interactions.apply(lambda row: (self.clusters[row['protein1']], self.clusters[row['protein2']]) 
                                                      if row['protein1'] < row['protein2'] 
                                                      else (self.clusters[row['protein2']], self.clusters[row['protein1']]), 
                                                      axis=1)

        if self.max_positive_pairs is not None:
            self.max_positive_pairs = min(self.max_positive_pairs, len(interactions))
            interactions = interactions.sort_values(by=['combined_score'], ascending=False).iloc[:self.max_positive_pairs]


        proteins_dict1 = interactions['protein1'].value_counts().to_dict()
        proteins_dict2 = interactions['protein2'].value_counts().to_dict()
        proteins_dict = {k: proteins_dict1.get(k, 0) + proteins_dict2.get(k, 0) for k in set(proteins_dict1) | set(proteins_dict2)}

        with open(self.SAVE_SEQ_PATH, 'w') as f:
            for id, seq in self.fasta_records.items():
                if id in proteins_dict:
                    f.write('>{}\n{}\n'.format(id, seq))

        logging.info('Final preprocessing for only positive pairs done.')
        logging.info('Sequences are saved to {}'.format(self.SAVE_SEQ_PATH))
        logging.info('Number of proteins: {}'.format(len(proteins_dict)))
        logging.info('Number of interactions: {}'.format(len(interactions)))
        
        return interactions, proteins_dict
    
    def negative_pairs_multiprocessing(self, proteins1, proteins2, unique_clusters, i):
        negative_pairs = []
        for j in range(i, i + 10000):
            p1 = proteins1[j]
            p2 = proteins2[j]
            sorted_pair = (p1, p2) if p1 < p2 else (p2, p1)
            cluster_pair = (self.clusters[sorted_pair[0]], self.clusters[sorted_pair[1]])
            if sorted_pair in negative_pairs or cluster_pair in unique_clusters:
                continue
            negative_pairs.append(sorted_pair)

        logging.info('Generated {} negative pairs.'.format(len(negative_pairs)))
        return negative_pairs


    def create_negatives(self, interactions, proteins_dict):
        
        positive_len = len(interactions)
        
        logging.info('Generating negative pairs.')
        logging.info(positive_len)

        # Convert the node degrees in protein dict into probabilities for negative sampling
        sum_proteins = sum(proteins_dict.values())
        proteins_dict = {k: v / sum_proteins for k, v in proteins_dict.items()}

        proteins1 = random.choices(list(proteins_dict.keys()), weights=proteins_dict.values(), k=positive_len * 15)
        proteins2 = random.choices(list(proteins_dict.keys()), weights=proteins_dict.values(), k=positive_len * 15)
        
        # negative_pairs = pd.DataFrame({'protein1': proteins1, 'protein2': proteins2, 'combined_score': 0})

        # logging.info('Negative pairs generated. Processing started.')
        
        # negative_pairs['protein1'], negative_pairs['protein2'] = zip(*negative_pairs.apply(
        #     lambda x: (x['protein1'], x['protein2']) if x['protein1'] < x['protein2'] else (
        #         x['protein2'], x['protein1']), axis=1))

        unique_clusters = set(interactions['clusters'].values)
        interactions = interactions.drop(columns='clusters')

        logging.info('Unique clusters: {}'.format(len(unique_clusters)))

        negative_pairs = []
        # i = 0
        # while len(negative_pairs) < positive_len * 10:
        #     # get a pair randomly (p1, p2) from the proteins_dict, each protein has a probability of being selected
        #     p1 = proteins1[i]
        #     p2 = proteins2[i]
        #     i += 1
        #     # p1 = random.choices(list(proteins_dict.keys()), weights=proteins_dict.values(), k=1)[0]
        #     # p2 = random.choices(list(proteins_dict.keys()), weights=proteins_dict.values(), k=1)[0]
        #     sorted_pair = (p1, p2) if p1 < p2 else (p2, p1)
        #     cluster_pair = (self.clusters[sorted_pair[0]], self.clusters[sorted_pair[1]])
        #     if sorted_pair in negative_pairs or cluster_pair in unique_clusters:
        #         continue
        #     negative_pairs.append(sorted_pair)

        #     if len(negative_pairs) % 10000 == 0:
        #         logging.info('Generated {} negative pairs.'.format(len(negative_pairs)))

        with multiprocessing.Pool(self.params.threads_per_worker) as pool:
            negative_pairs = pool.starmap(self.negative_pairs_multiprocessing, [(proteins1, proteins2, unique_clusters, i) for i in range(0, positive_len * 15, 10000)])
            negative_pairs = [item for sublist in negative_pairs for item in sublist]
            negative_pairs = negative_pairs[:positive_len * 10]
        
        negative_pairs = pd.DataFrame(negative_pairs, columns=['protein1', 'protein2'])
        negative_pairs['combined_score'] = 0

        # logging.info('Filtering out pairs that are in the same cluster with proteins interacting with a given one already.')

        # negative_pairs = negative_pairs[~negative_pairs.apply(
        #     lambda x: (self.clusters[x['protein1']], self.clusters[x['protein2']]) in unique_clusters, 
        #     axis=1)]

        logging.info('Removing duplicates from the dataset.')
        interactions = pd.concat([interactions, negative_pairs], ignore_index=True)
        interactions = interactions.drop_duplicates(subset=['protein1', 'protein2'], keep='first')

        if len(interactions) > positive_len * 11:
            logging.warning('Not enough negative pairs generated to suffice 1:10 positive-to-negative ratio.')
        
        interactions = interactions.iloc[:positive_len * 11]

        logging.info('Negative pairs generated. Saving to file.')
        
        interactions.to_csv(self.SAVE_PAIRS_PATH, sep='\t',
                            index=False,
                            header=False)
        
        return interactions
    

    def train_test_split(self, interactions, test_size=0.1):
        logging.info('Splitting into train and test sets.')

        interactions = interactions.sample(frac=1).reset_index(drop=True)
        
        test_size = int(len(interactions) * test_size)
        test_set = interactions.iloc[:test_size]
        train_set = interactions.iloc[test_size:]

        test_set.to_csv('.'.join(self.SAVE_PAIRS_PATH.split('.')[:-1]) + '_test.tsv', sep='\t', index=False, header=False)
        train_set.to_csv('.'.join(self.SAVE_PAIRS_PATH.split('.')[:-1]) + '_train.tsv', sep='\t', index=False, header=False)

        logging.info('Train and test sets saved to files.')



    # A method to remove sequences of inappropriate length from a fasta file
    def preprocess_fasta_file(self):
        logging.info('Getting protein records out of fasta file.')
        fasta_records ={record.id: record.seq for record in tqdm(SeqIO.parse(self.sequences_file, 'fasta'))}
        logging.info('Total number of proteins in the fasta file: {}'.format(len(fasta_records)))

        if not params.not_remove_long_short_proteins:
            logging.info(
                'Removing proteins that are shorter than {}aa or longer than {}aa.'.format(self.min_length,
                                                                                        self.max_length))

            fasta_records = {k: v for k, v in fasta_records.items() if
                                    self.min_length <= len(v) <= self.max_length}
            
            logging.info('Total number of proteins after filtering: {}'.format(len(fasta_records)))

        return fasta_records

def add_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--species", type=str, default=None,
                        help="The Taxon identifier of the organism of interest.")
    group.add_argument("--int_seq", nargs=2, metavar=("interactions", "sequences"), default=None,
                       help="The physical links (full) file from STRING and the fasta file with sequences. "
                            "Two paths should be separated by a whitespace. "
                            "If not provided, they will be downloaded from STRING. For both files see "
                            "https://string-db.org/cgi/download")
    parser.add_argument("--not_remove_long_short_proteins", action='store_true',
                        help="If specified, does not remove proteins "
                             "shorter than --min_length and longer than --max_length. "
                             "By default, long and short proteins are removed.")
    parser.add_argument("--min_length", type=int, default=50,
                        help="The minimum length of a protein to be included in the dataset.")
    parser.add_argument("--max_length", type=int, default=800,
                        help="The maximum length of a protein to be included in the dataset.")
    parser.add_argument("--max_positive_pairs", type=int, default=None,
                        help="The maximum number of positive pairs to be included in the dataset. "
                             "If None, all pairs are included. If specified, the pairs are selected "
                             "based on the combined score in STRING.")
    parser.add_argument("--combined_score", type=int, default=500,
                        help="The combined score threshold for the pairs extracted from STRING. "
                             "Ranges from 0 to 1000.")
    parser.add_argument("--experimental_score", type=int, default=None,
                        help="The experimental score threshold for the pairs extracted from STRING. "
                             "Ranges from 0 to 1000. Default is None, which means that the experimental "
                             "score is not used.")
    parser.add_argument("--n_workers", type=int, default=multiprocessing.cpu_count(),
                        help="The number of workers to use for parallel processing.")
    parser.add_argument("--threads_per_worker", type=int, default=1,
                        help="The number of threads per worker.")
    parser.add_argument("--memory_limit", type=str, default='4GB',
                        help="The memory limit for each worker.")

    return parser


def main(params):
    downloaded_flag = False
    params.interactions = params.int_seq[0] if params.int_seq is not None else None
    params.sequences = params.int_seq[1] if params.int_seq is not None else None
    if params.species is not None:
        downloaded_flag = True
        logging.info('One or both of the files are not specified (interactions or sequences). '
                     'Downloading from STRING...')

        _, version = get_string_url()
        logging.info('STRING version: {}'.format(version))

        try:
            url = "{0}protein.physical.links.full.v{1}/{2}.protein.physical.links.full.v{1}.txt.gz".format(
                DOWNLOAD_LINK_STRING, version, params.species)
            string_file_name_links = "{1}.protein.physical.links.full.v{0}.txt".format(version, params.species)
            wget.download(url, out=string_file_name_links + '.gz')
            with gzip.open(string_file_name_links + '.gz', 'rb') as f_in:
                with open(string_file_name_links, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            url = "{0}protein.sequences.v{1}/{2}.protein.sequences.v{1}.fa.gz".format(DOWNLOAD_LINK_STRING, version,
                                                                                      params.species)
            string_file_name_seqs = "{1}.protein.sequences.v{0}.fa".format(version, params.species)
            wget.download(url, out=string_file_name_seqs + '.gz')
            with gzip.open(string_file_name_seqs + '.gz', 'rb') as f_in:
                with open(string_file_name_seqs, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except HTTPError:
            raise Exception('The files are not available for the specified species. '
                            'There might be two reasons for that: \n '
                            '1) the species is not available in STRING. Please check the STRING species list to '
                            'verify. \n '
                            '2) the download link has changed. Please raise an issue in the repository. ')

        os.remove(string_file_name_seqs + '.gz')
        os.remove(string_file_name_links + '.gz')

        params.interactions = string_file_name_links
        params.sequences = string_file_name_seqs

    data = STRINGDatasetCreation(params)

    if downloaded_flag:
        os.remove(params.interactions)
        os.remove(params.sequences)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if '/home/volzhenin/mmseqs/bin/' not in os.environ['PATH']:
        os.environ['PATH'] = '/home/volzhenin/mmseqs/bin/:' + os.environ['PATH']

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    params = parser.parse_args()
    main(params)

    # data = pd.read_csv('protein.pairs_23.tsv', sep='\t', header=None, names=['protein1', 'protein2', 'combined_score'])
    # data_negative = data[data['combined_score'] == 0]
    # data_positive = data[data['combined_score'] == 1]

    # data_positive_deg1 = data_positive['protein1'].value_counts().to_dict()
    # data_positive_deg2 = data_positive['protein2'].value_counts().to_dict()
    # data_positive_deg = {k: data_positive_deg1.get(k, 0) + data_positive_deg2.get(k, 0) for k in set(data_positive_deg1) | set(data_positive_deg2)}

    # data_negative_deg1 = data_negative['protein1'].value_counts().to_dict()
    # data_negative_deg2 = data_negative['protein2'].value_counts().to_dict()
    # data_negative_deg = {k: data_negative_deg1.get(k, 0) + data_negative_deg2.get(k, 0) for k in set(data_negative_deg1) | set(data_negative_deg2)}

    # #plot two node degree distributions
    # import matplotlib.pyplot as plt
    # import numpy as np

    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].hist(list(data_positive_deg.values()), bins=100, color='blue', alpha=0.7, label='Positive pairs')
    # ax[0].set_title('Positive pairs')
    # ax[0].set_xlabel('Node degree')
    # ax[0].set_ylabel('Frequency')
    
    # ax[1].hist(list(data_negative_deg.values()), bins=100, color='red', alpha=0.7, label='Negative pairs')
    # ax[1].set_title('Negative pairs')
    # ax[1].set_xlabel('Node degree')
    # ax[1].set_ylabel('Frequency')

    # plt.savefig('node_degrees.png')

