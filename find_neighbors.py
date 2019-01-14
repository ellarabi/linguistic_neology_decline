import os
import csv
import pickle

from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
		serialization of an object
		:param obj: object to serialize
		:param name: file name to store the object
		"""
        with open('pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
		de-serialization of an object
		:param name: file name to load the object from
		"""
        with open('pickle/' + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)
        # end with
    # end def
# end class


class EstimateSemanticDensity:
    def __init__(self):
        self._word_pairs = {}
    # end def

    @staticmethod
    def convert_to_word2vec_format(embeddings, dirname):
        print('loading embeddings...')
        outfile = dirname + 'gensim.glove.vectors.txt'
        glove2word2vec(glove_input_file=embeddings, word2vec_output_file=outfile)
        print('converted glove to word2vec embeddings...')
        return outfile
    # end def

    def _read_parallel_word_list(self, filename):
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = next(csv_reader)  # skip header
            for line in csv_reader:
                if len(line) < 2: continue
                self._word_pairs[line[0].strip()] = line[1].strip()
            # end for
        # end with
    # end def

    @staticmethod
    def _compute_avg_similarities(neighbors, num_neighbors):
        similarities = []
        for num in range(5, num_neighbors, 5):
            total_similarity = 0
            for i in range(0, num):
                total_similarity += neighbors[i][1]
            # end for
            similarities.append(float(total_similarity)/num)
        # end for
        return similarities
    # end def

    @staticmethod
    def _compute_neighbors_in_radius(neighbors):
        import numpy
        neighbors_count = []
        for radius in numpy.arange(0.55, 0.35, -0.025):
            count = 0
            for word in neighbors:
                if float(word[1]) >= radius:
                    count += 1
                # end if
            # end for
            neighbors_count.append(count)
        # end for
        return neighbors_count
    # end def

    def compute_neighborhood_density(self, vectors, corr_filename, type='dist'):
        model_filename = 'glove.embeddings'
        if os.path.isfile('pickle/' + model_filename + '.pkl'):
            model = Serialization.load_obj(model_filename)
        else:
            model = KeyedVectors.load_word2vec_format(vectors, binary=False)
            print('loaded word2vec-like embeddings, saving model...')
            Serialization.save_obj(model, model_filename)
        # end if

        assert type == 'dist' or type == 'radius'

        self._read_parallel_word_list(corr_filename)
        print('loaded', len(self._word_pairs), 'word pairs')

        num_neighbors = 50 if type == 'dist' else 5000
        outfile = 'cos.similarities.dist.tsv' if type == 'dist' else 'cos.similarities.radius.tsv'

        with open(outfile, 'w') as fout:
            for word in self._word_pairs.keys():
                try:
                    org_neighbors = model.most_similar(word, topn=num_neighbors)
                    alt_neighbors = model.most_similar(self._word_pairs[word], topn=num_neighbors)
                except:
                    print(word, 'not found in embeddings')
                    continue
                # end try

                if type == 'dist':
                    org_avg_similarities = self._compute_avg_similarities(org_neighbors, num_neighbors)
                    alt_avg_similarities = self._compute_avg_similarities(alt_neighbors, num_neighbors)
                else:
                    org_avg_similarities = self._compute_neighbors_in_radius(org_neighbors)
                    alt_avg_similarities = self._compute_neighbors_in_radius(alt_neighbors)
                # end if

                outstr = word + '\t' + '\t'.join(list(map(str, org_avg_similarities)))
                outstr += ('\t' + self._word_pairs[word] + '\t' +
                           '\t'.join(list(map(str, alt_avg_similarities))))
                fout.write(outstr + '\n')
                fout.flush()
            # end for
        # end with

    #end def

# end class


if __name__ == '__main__':
    dirname = '/ais/hal9000/ella/embeddings/'
    #filename = dirname + 'glove.42B.300d.txt'  # embeddings file

    dens_estimator = EstimateSemanticDensity()
    #outfile = dens_estimator.convert_to_word2vec_format(filename, dirname)

    vectors = dirname + 'gensim.glove.vectors.txt'
    pairs_filename = '/u/ella/projects/language_neology/decline.stable.pairs.csv'
    dens_estimator.compute_neighborhood_density(vectors, pairs_filename, type='radius') # 'dist' or 'radius'

    # distance or radius file can now be analysed (excel)


# end if
