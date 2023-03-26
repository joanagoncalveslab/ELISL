from src import config
import networkx as nx
from node2vec import Node2Vec
import src.data_functions as dfnc
import gensim

PROJECT_LOC = config.ROOT_DIR


def get_save_loc(opts, prefix='embs', ppi_cutoff='0.0', folder_loc='embeddings/node2vec', source='e'):
    if prefix=='embs':
        emb_name = f'{prefix}_ppi_{source}={ppi_cutoff}_' \
                   f'dim={opts["dimensions"]}_' \
                   f'p={opts["p"]}_' \
                   f'q={opts["q"]}_' \
                   f'wl={opts["walk_length"]}_' \
                   f'nw={opts["num_walks"]}_' \
                   f'ws={opts["workers"]}_' \
                   f'wd={opts["window"]}_' \
                   f'mc={opts["min_count"]}_' \
                   f'bw={opts["batch_words"]}'
        save_loc = config.DATA_DIR / folder_loc / emb_name
    if prefix=='model':
        emb_name = f'{prefix}_ppi_{source}={ppi_cutoff}_' \
                   f'dim={opts["dimensions"]}_' \
                   f'p={opts["p"]}_' \
                   f'q={opts["q"]}_' \
                   f'wl={opts["walk_length"]}_' \
                   f'nw={opts["num_walks"]}_' \
                   f'ws={opts["workers"]}'
        save_loc = config.DATA_DIR / folder_loc / emb_name

    return save_loc


def get_node2vec_emb(ppi_graph, ppi_cutoff='0.0', folder_loc='embeddings/node2vec', opts_extra={},
                     source='e'):
    opts = {'dimensions': 64, 'walk_length': 30, 'num_walks': 200, 'p': 1, 'q': 1, 'workers': 4,
            'window': 10, 'min_count': 1, 'batch_words': 4}
    opts.update(opts_extra)

    node2vec = Node2Vec(ppi_graph, dimensions=opts['dimensions'], p=opts['p'], q=opts['q'],
                        walk_length=opts['walk_length'], num_walks=opts['num_walks'], workers=opts['workers'])
    model = node2vec.fit(window=opts['window'], min_count=opts['min_count'], batch_words=opts['batch_words'])

    # model.wv.most_similar('2')
    embed_loc = get_save_loc(opts, 'embs',ppi_cutoff, folder_loc, source)
    model_loc = get_save_loc(opts, 'model',ppi_cutoff, folder_loc, source)

    # Save embeddings
    model.wv.save_word2vec_format(embed_loc)

    # Save model
    model.save(str(model_loc)+'.model')

    # Embed edges by Hadamard method
    #edges_emb = HadamardEmbedder(keyed_vectors=model.wv)
    #edges_kv = edges_emb.as_keyed_vectors()


def load_embs(ppi_cutoff='0.0', folder_loc='embeddings/node2vec',
              opts_extra={}, source='e'):
    opts = {'dimensions': 64, 'walk_length': 30, 'num_walks': 200, 'p': 1, 'q': 1, 'workers': 4,
            'window': 10, 'min_count': 1, 'batch_words': 4}
    opts.update(opts_extra)
    embed_loc = get_save_loc(opts, 'embs',ppi_cutoff, folder_loc, source=source)
    loaded_vectors = gensim.models.KeyedVectors.load_word2vec_format(embed_loc)
    print(f'Embeddings at {embed_loc} are loaded')
    return loaded_vectors


def load_embs_from_raw_loc(embed_loc=''):
    loaded_vectors = gensim.models.KeyedVectors.load_word2vec_format(embed_loc)
    print(f'Embeddings at {embed_loc} are loaded')
    return loaded_vectors


def get_embs_dict(w2v_vectors):
    return dict(zip(w2v_vectors.index2word, w2v_vectors.syn0))


def load_model(ppi_cutoff='0.0', folder_loc='embeddings/node2vec', opts_extra={}, source='e'):
    opts = {'dimensions': 64, 'walk_length': 30, 'num_walks': 200, 'p': 1, 'q': 1, 'workers': 4,
          'window': 10, 'min_count': 1, 'batch_words': 4}
    opts.update(opts_extra)
    model_loc = get_save_loc(opts, 'model',ppi_cutoff, folder_loc, source=source)
    loaded_model = gensim.models.Word2Vec.load(str(model_loc) + '.model')
    return loaded_model

def main():
    cut_off_df, full_df = dfnc.get_PPI_data(cutoff_list=[0.0], source='ec')
    cut_off = '0.0'
    ppi_graph = nx.from_pandas_edgelist(cut_off_df[cut_off], 'protein1', 'protein2').to_undirected()
    # graph = nx.fast_gnp_random_graph(n=100, p=0.5)
    for p in [1]:#, 2, 4, 8]:
        for q in [1]:#0.25, 0.5, 0.75, 1, 2, 4, 8]:
            opts = {'dimensions': 64, 'walk_length': 30, 'num_walks': 200, 'p': p, 'q': q, 'workers': 4,
                    'window': 10, 'min_count': 1, 'batch_words': 4}
            get_node2vec_emb(ppi_graph, ppi_cutoff=cut_off, opts_extra=opts, source='ec')


if __name__ == '__main__':
    main()
