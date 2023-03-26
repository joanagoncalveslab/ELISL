from src import config
import collections as coll
import pandas as pd
import numpy as np
import json
#from allennlp.commands.elmo import ElmoEmbedder
from sklearn.decomposition import PCA


PROJECT_LOC = config.ROOT_DIR


#Via the seqvec pip package, you can compute embeddings for a fasta file with the seqvec command. Add --protein True to get an embedding per sequence instead of per residue.
#seqvec -i sequences.fasta -o embeddings.npz
#If you specify .npy as output format (e.g. with -o embeddings.npy), the script will save the embeddings as an numpy array and the corresponding identifiers (as extracted from the header line in the fasta file) in a json file besides it. The sorting in the json file corresponds to the indexing in the npy file. The npy file can be loaded via:
#Load the embeddings with numpy:
def load_embeddings(emb_loc="embeddings.npz"):
    emb_loc = config.DATA_DIR / emb_loc
    if str(emb_loc).split('.')[-1] == "npz":
        data = np.load(emb_loc) #type: Dict[str, np.ndarray]
        data = dict(data)
    elif str(emb_loc).split('.')[-1] == "npy":
        embs = np.load("embeddings.npy") # shape=(n_proteins,)
        with open("embeddings.json") as fp:
            labels = json.load(fp)
        data = dict(zip(labels, embs))
    data = coll.OrderedDict(sorted(data.items()))
    return data


def apply_pca(in_loc='embeddings/seqvec/uniprot_1024_embeddings.npz',
                      n_components=None, random_state=124, save_loc=''):
    emb_dict = load_embeddings(in_loc)
    df = pd.DataFrame.from_dict(emb_dict, orient='index')
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_values = pca.fit_transform(df.values)
    reduced_df = pd.DataFrame(index=df.index, data=reduced_values)
    reduced_dict = coll.OrderedDict(zip(reduced_df.index.values,reduced_df.values))
    if save_loc != '':
        save_loc = config.DATA_DIR / save_loc
        np.savez(save_loc, **reduced_dict)

    return reduced_dict



#Load pre-trained model:
def get_pretrained_model(weight_loc="seqvec/pretrained/weights.hdf5", options_loc="seqvec/pretrained/options.json"):
    weights = config.DATA_DIR / weight_loc
    options = config.DATA_DIR / options_loc
    embedder = ElmoEmbedder(options,weights, cuda_device=0) # cuda_device=-1 for CPU


#Get embedding for amino acid sequence:
def get_seq_embs(seq, embedder):
    seq = 'SEQWENCE' # your amino acid sequence
    embedding = embedder.embed_sentence(list(seq)) # List-of-Lists with shape [3,L,1024]


#Get embeddings for multi aa sequences
def get_seqs_embs(seqs, embedder):
    #seq1 = 'SEQWENCE' # your amino acid sequence
    #seq2 = 'PROTEIN'
    #seqs = [list(seq1), list(seq2)]
    #seqs.sort(key=len) # sorting is crucial for speed
    embedding = embedder.embed_sentences(seqs) # returns: List-of-Lists with shape [3,L,1024]

    return embedding


#Get 1024-dimensional embedding for per-residue predictions:
def get_emb_per_residue(embedding):
    import torch
    residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]


#Get 1024-dimensional embedding for per-protein predictions:
def get_emb_per_prot(embedding):
    import torch
    protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]


def main():
    #ppi_fasta_seqs = dfnc.get_PPI_fasta_seq()
    #cut_off_df, full_df = dfnc.get_PPI_data(cutoff_list=[0.0])
    #ppi_graph = nx.from_pandas_edgelist(cut_off_df['0.0'], 'protein1', 'protein2').to_undirected()
    #graph = nx.fast_gnp_random_graph(n=100, p=0.5)
    emb_loc = config.DATA_DIR / 'embeddings' / 'seqvec' / 'uniprot_1024_embeddings.npz'
    emb_dict = load_embeddings(emb_loc)
    n_comp = 16
    for n_comp in [16,32,64,128,256,512]:
        saving = 'embeddings/seqvec/uniprot_'+str(n_comp)+'_embeddings.npz'
        apply_pca(n_components=n_comp, save_loc=saving)

    print(0)

if __name__ == '__main__':
    main()
