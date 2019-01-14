import os
import nltk
import pickle
import glob
import math

from scipy import stats
from find_neighbors import Serialization


class Utils:
    @staticmethod
    def load_vocabulary(filename):
        """
        extract vocabulary of nouns
        :param filename:
        :return:
        """
        vocab_filename = 'pickle/vocabulary.pkl'
        if os.path.isfile(vocab_filename):
            return Serialization.load_obj('vocabulary')
        # end if

        vocabulary = {}
        with open(filename, 'r') as fin:
            for line in fin:
                tokens = line.strip().split()
                if len(tokens) < 4 or not tokens[0].isdigit(): continue

                word = tokens[1].lower()
                pos_tag = nltk.tag.pos_tag([word])[0][1]
                if pos_tag != 'NN': continue
                vocabulary[word] = ''
            # end for
        # end with
        print('extracted', len(vocabulary), 'nouns')
        Serialization.save_obj(vocabulary, 'vocabulary')
        return vocabulary

    # end def

    @staticmethod
    def normalize(d, total):
        """
        raw count to frequency
        :param d:
        :param total:
        :return:
        """
        dn = {}
        for key in d.keys():
            dn[key] = float(d[key]) / total
        # end for
        return dn
    # end def

# end class


class WordsTrendExtractor():
    def __init__(self):
        self._years_dict = {}
    # end def

    def extract_word_frequencies(self, dirname, vocabulary):
        freq_filename = 'frequency.by.years'
        if os.path.isfile('pickle/' + freq_filename + '.pkl'):
            self._years_dict = Serialization.load_obj(freq_filename)
            return
        # end if

        text_dirname = dirname + 'COHA_text/'
        subdirs = ['1810s', '1820s', '1830s', '1840s', '1850s',
                   '1860s', '1870s', '1880s', '1890s', '1900s',
                   '1910s', '1920s', '1930s', '1940s', '1950s',
                   '1960s', '1970s', '1980s', '1990s', '2000s']

        for subdir in subdirs:
            current_dir = text_dirname + subdir + '/'
            print('processing', current_dir)

            total_tokens = 0
            word_frequencies = {}
            for filename in glob.glob(current_dir + '*.txt'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        tokens = line.split()
                        total_tokens += len(tokens)
                        for token in tokens:
                            # we only consider nouns from vocabulary
                            if vocabulary.get(token, None) is None: continue
                            count = word_frequencies.get(token.lower(), 0)
                            word_frequencies[token.lower()] = count + 1
                        # end for
                    # end for
                # end with
            # end for
            self._years_dict[subdir] = Utils.normalize(word_frequencies, total_tokens)
        # end for
        Serialization.save_obj(self._years_dict, freq_filename)
    # end def

    def estimate_word_trend(self, vocabulary, outfile):
        time_steps = []
        for i in list(reversed(range(1, 21))):
            time_steps.append(i)
        # end for
        #print(time_steps)

        with open(outfile, 'w') as fout:
            for word in vocabulary:
                word_freq_array = []
                for key in sorted(self._years_dict.keys()):
                    frequency = self._years_dict[key].get(word, 0)
                    word_freq_array.append(math.log(frequency) if frequency > 0 else 0)
                    #word_freq_array.append(frequency)
                # end for
                assert(len(time_steps) == len(word_freq_array))

                corr, pval = stats.spearmanr(time_steps, word_freq_array)
                fout.write(word + '\t' + str(corr) + '\t' +
                           str(pval) + '\n')
            # end for
    # end def

    def fetch_word_trend(self, word):
        word_freq_array = []
        for key in sorted(self._years_dict.keys()):
            frequency = self._years_dict[key].get(word, 0)
            # word_freq_array.append(math.log(frequency) if frequency > 0 else 0)
            word_freq_array.append(frequency)
        # end for
        print(word_freq_array)
    # end def

    def pair_words_for_comparison(self, filename):
        decline = []; stable = []
        frequencies_beg = self._years_dict['1810s']
        frequencies_end = self._years_dict['2000s']

        with open(filename, 'r') as fin:
            for line in fin:
                tokens = line.strip().split()
                if len(tokens) < 3 or tokens[1] == 'nan': continue

                freq_beg = frequencies_beg.get(tokens[0], 0)
                freq_end = frequencies_end.get(tokens[0], 0)
                if not tokens[0].isalpha(): continue

                if float(tokens[1]) > DECLINE_SPEARMANS_CORRELATION:
                    if freq_end == 0 or float(freq_beg) / freq_end > MIN_FREQUENCY_RATIO:
                        decline.append(tokens[0])
                    # end if
                # end if

                if math.fabs(float(tokens[1])) < STABLE_SPEARMANS_CORRELATION:
                    stable.append(tokens[0])
                # end if
            # end for
        # end with
        print('found', len(decline), 'decline words')
        print('found', len(stable),  'stable words ')

        pairs_dict = {}
        for dword in decline:
            dword_len = len(dword)
            dword_freq = frequencies_end.get(dword, 0)

            found = False
            for sword in stable:
                sword_len = len(sword)
                sword_freq = frequencies_end.get(sword, 0)
                if sword_freq == 0: continue

                if abs(dword_len-sword_len) < 2:
                    ratio = float(dword_freq)/sword_freq
                    if ratio > 0.75 and ratio < 1.33:
                        pairs_dict[dword] = sword
                        stable.remove(sword)
                        found = True
                        break
                    # end if
                # end if
            # end for

            if not found:
                print('not matched:', dword)
            # end if
        # end for
        return pairs_dict
    # end def

# end class


MIN_FREQUENCY_RATIO = 5
DECLINE_SPEARMANS_CORRELATION = 0.9
STABLE_SPEARMANS_CORRELATION  = 0.1

if __name__ == '__main__':
    dirname = '/ais/hal9000/ella/language_neology/COHA/'
    vocabulary = Utils.load_vocabulary(dirname + 'lexicon.nn.txt')
    print('loaded vocabulary with', len(vocabulary), 'nouns')


    extractor = WordsTrendExtractor()
    extractor.extract_word_frequencies(dirname, vocabulary)
    #extractor.estimate_word_trend(vocabulary, 'correlation.log.dat')

    pairs_dict = extractor.pair_words_for_comparison('correlation.log.dat')
    with open('decline.stable.pairs.csv', 'w') as fout:
        for dword in pairs_dict.keys():
            fout.write(','.join([dword, pairs_dict[dword]]) + '\n')
        # end for
    # end with

    #extractor.fetch_word_trend('kid')

# end if
