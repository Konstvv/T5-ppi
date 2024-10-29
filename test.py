import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import PairSequenceData, SequencesDataset
import os

def pred_model(model, seq1, seq2):

    model.eval()

    batch = {'ids1': [seq1], 'ids2': [seq2]}

    with torch.no_grad():
        pred, (attn1, attn2) = model(batch, return_attention=True)

    print(pred, attn1, attn2)
    attn1 = attn1.mean(-1).squeeze()
    attn2 = attn2.mean(-1).squeeze()

    return pred, (attn1, attn2), (seq1, seq2)

def test_model(model, dataset):

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto',
                          devices=params.devices, 
                          num_nodes=params.num_nodes)

    pred_loader = DataLoader(dataset=dataset, batch_size=params.batch_size, num_workers=8, shuffle=False, collate_fn=dataset.collate_fn)
    trainer.test(model, pred_loader)


from Bio.PDB import PDBParser, NeighborSearch, is_aa

def get_interaction_map(pdb_file, chain_a_id='A', chain_b_id='B', distance_cutoff=5.0):
    """
    Extracts the positions in Chain A where it interacts with Chain B based on atomic distances.
    
    Parameters:
    pdb_file (str): Path to the PDB file.
    chain_a_id (str): Chain ID for Chain A. Default is 'A'.
    chain_b_id (str): Chain ID for Chain B. Default is 'B'.
    distance_cutoff (float): Distance cutoff to consider interaction (default is 5.0 Å).
    
    Returns:
    list: A list of 0s and 1s, where 1 indicates interaction at that residue position in Chain A.
    """
    # Initialize PDB parser
    parser = PDBParser(QUIET=True)
    
    # Load the structure from the PDB file
    structure = parser.get_structure('PDB_ID', pdb_file)

    # Select the model (assuming model 0 for simplicity)
    model = structure[0]

    # Extract chains A and B
    chain_A = model[chain_a_id]
    chain_B = model[chain_b_id]

    # Get all atoms from chain A and chain B
    atoms_chain_A = [atom for atom in chain_A.get_atoms() if is_aa(atom.get_parent())]
    atoms_chain_B = [atom for atom in chain_B.get_atoms() if is_aa(atom.get_parent())]

    # Create a NeighborSearch object for chain B atoms
    neighbor_search = NeighborSearch(atoms_chain_B)

    # Create a dictionary to store residue interactions
    interaction_map_A = {}
    
    # Find interacting residues
    for atom in atoms_chain_A:
        close_atoms = neighbor_search.search(atom.coord, distance_cutoff)
        if close_atoms:  # If any atom from chain B is within the cutoff distance
            residue_id = atom.get_parent().get_id()[1]  # Get the actual residue number (not zero-indexed)
            interaction_map_A[residue_id] = 1  # Mark the position as interacting

    # Get all residue IDs in chain A and create a map list
    residue_ids_A = [residue.get_id()[1] for residue in chain_A.get_residues() if is_aa(residue)]
    
    # Create a list initialized with zeros of length equal to the number of residues in Chain A
    full_interaction_map_A = [0] * len(residue_ids_A)

    # Mark the positions of interacting residues with 1
    for i, residue_id in enumerate(residue_ids_A):
        if residue_id in interaction_map_A:
            full_interaction_map_A[i] = 1

    # Return the full interaction map
    return full_interaction_map_A

# Example usage:
# interaction_map = get_interaction_map('path_to_pdb_file.pdb')
# print(interaction_map)



if __name__ == '__main__':
    # print('Loading interaction map...')
    # int_map = get_interaction_map('7ypd.pdb')

    from model_esm import PPITransformerModel
    print('Loading model...')

    # tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenizer')

    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()
    params.max_len = 1000
    params.batch_size = 32
    
    # model = PPITransformerModel(params, 
    #                 ntoken=len(tokenizer), 
    #                 embed_dim=512,#64, #512
    #                 hidden_dim=2048, #512 #2048
    #                 num_siamese_layers=12, #6
    #                 num_cross_layers=6, #3
    #                 num_heads=8, #8
    #                 dropout=0.1)
    # checkpoint_folder = os.path.join('ppi-transformer/59lrxzqy/', 'checkpoints')

    model = PPITransformerModel(params, 
                           embed_dim=480,
                            hidden_dim=2048,
                            num_cross_layers=6,
                            num_heads=8,
                            dropout=0.1)
    
    #checkpoint_folder = os.path.join('ppi-transformer/tmg60lyz', 'checkpoints')
    checkpoint_folder = os.path.join('ppi-transformer/sintk29b', 'checkpoints')
    # checkpoint_folder = os.path.join('ppi-transformer/dpk8ayx7', 'checkpoints')

    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    print('Loading model from checkpoint:', checkpoint_path)
    if torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path, weights_only=False)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # for key in list(ckpt['state_dict'].keys()):
    #     if key.startswith('embedding'):
    #         del ckpt['state_dict'][key]
    # model.load_state_dict(ckpt['state_dict'])

    model_state_dict = model.state_dict()
    model_state_dict.update(ckpt['state_dict'])
    model.load_state_dict(model_state_dict)

    sequences = SequencesDataset(sequences_path="aphid_dev.fasta")

    dataset = PairSequenceData(pairs_path="aphid_dev.tsv",
                                sequences_dataset=sequences,
                                # tokenizer=tokenizer, 
                                for_esm=True)
    
    # test_model(checkpoint_folder='logs/AttentionModelBase/version_173', dataset=dataset)
    test_model(model, dataset)

    # seq1 = 'MTPSATVYVISGATRGIGFALTSLLAQRDNVLIFAGARDPAKALPLQALAAATGKVIPVKLESANEEDAAALAKLVKEKAGKVDFLLANAGVCELNKPVLSTPSATFVDHFTVNTLGPLTLFQQFYSLLTESSSPRFFVTSSAGGSTTYVSMAPDMDLAPAYGISKAAVNHLVAHIARKFGAKDGLVAAVVHPGLVATDMTRPFLEAAGLPADGGPGFEHISPDESAAALVKIFDEAKRETHGGKFLSYDGTEIPW'
    # seq2 = 'MTPSATVYVISGATRGIGFALTSLLAQRDNVLIFAGARDPAKALPLQALAAATGKVIPVKLESANEEDAAALAKLVKEKAGKVDFLLANAGVCELNKPVLSTPSATFVDHFTVNTLGPLTLFQQFYSLLTESSSPRFFVTSSAGGSTTYVSMAPDMDLAPAYGISKAAVNHLVAHIARKFGAKDGLVAAVVHPGLVATDMTRPFLEAAGLPADGGPGFEHISPDESAAALVKIFDEAKRETHGGKFLSYDGTEIPW'

    # print(seq1)
    # print(seq2)

    # pred, (attn1, attn2), (tokens_seq1, tokens_seq2) = pred_model(model, seq1, seq2)

    # print(pred)

    # from matplotlib import pyplot as plt
    # import numpy as np
    # from mpl_toolkits.mplot3d import Axes3D

    # fig, ax = plt.subplots(1, 2, figsize=(60, 40))

    # attn1 = attn1.numpy()
    # attn1[attn1 == 0] = np.nan
    # ax[0].imshow(attn1)
    # # the first dimantion corresponds to tokens_seq1, the second to tokens_seq2
    # ax[0].set_xticks(range(len(tokens_seq2)))
    # ax[0].set_xticklabels(tokens_seq2)
    # ax[0].set_yticks(range(len(tokens_seq1)))
    # ax[0].set_yticklabels(tokens_seq1)

    # attn2 = attn2.numpy()
    # attn2[attn2 == 0] = np.nan
    # ax[1].imshow(attn2)
    # ax[1].set_xticks(range(len(tokens_seq1)))
    # ax[1].set_xticklabels(tokens_seq1)
    # ax[1].set_yticks(range(len(tokens_seq2)))
    # ax[1].set_yticklabels(tokens_seq2)

    # plt.savefig('attn.pdf')

    # #use attn1 and plot the max value along both dimentions, create a figure with 2 plots

    # attn_x = attn1.max(0)
    # attn_y = attn1.max(1)

    # fig, ax = plt.subplots(1, 2, figsize=(60, 5))

    # ax[0].plot(attn_x)
    # ax[1].plot(attn_y)

    # ax[0].set_xticks(range(len(tokens_seq2)))
    # ax[0].set_xticklabels(tokens_seq2)
    # ax[1].set_xticks(range(len(tokens_seq1)))
    # ax[1].set_xticklabels(tokens_seq1)

    # #plot interaction map on ax[0]

    # ax[0].plot(int_map)
    # print(int_map)
    # print(len(int_map))
    # print(max(int_map))
    # print(len(attn_x))

    # plt.savefig('attn_max.pdf')