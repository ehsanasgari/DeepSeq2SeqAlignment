from __future__ import unicode_literals, print_function, division
import unicodedata
import re
import random
import numpy as np
from torch.autograd import Variable
import torch
import nltk
import itertools

use_cuda = torch.cuda.is_available()

class SequenceReader(object):
    '''
        Sequence Reader Class
    '''
    PAD_token = 0
    GO_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1:"<GO>", 2: "<EOS>", 3:"<UNK>"}
        self.n_words = 4  # Count PAD, GO, EOS, and UNK

    def add_sequences(self, sequences):
        for sequence in sequences:
            self.add_sequences(sequence)

    def add_sequence(self, sequence, normalize=False, normalization_func=None):
        if normalize:
            if normalization_func:
                sequence = normalization_func(sequence)
            else:
                sequence = SequenceReader.normalizeString(SequenceReader.unicodeToAscii(sequence)).strip()
        for word in sequence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            if self.word2index[word] > 3:
                self.word2count[word] += 1

    def index_Sequence(self, sequence):
        return [self.word2index[word] if word in self.word2index else SequenceReader.UNK_token for word in sequence.split(' ')]

    def rep_sequence(self, sequence):
        indexes = self.index_Sequence(sequence)
        indexes.append(SequenceReader.EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return result.cuda()
        else:
            return result


    @staticmethod
    def preprocess_pairs(sequence_pairs, vocab_size, preprocess=False):
        if preprocess:
            input_symbols=[[SequenceReader.normalizeString(SequenceReader.unicodeToAscii(pair[0])).strip().split()] for pair in  sequence_pairs]
            output_symbols=[[SequenceReader.normalizeString(SequenceReader.unicodeToAscii(pair[1])).strip().split()] for pair in  sequence_pairs]
        else:
            input_symbols=[pair[0].split() for pair in  sequence_pairs]
            output_symbols=[pair[1].split() for pair in  sequence_pairs]

        # Frequency distribution of input
        freq_dist = nltk.FreqDist(itertools.chain(*input_symbols))
        input_vocab = [x for x,c in freq_dist.most_common(vocab_size)]

        # Frequency distribution of input
        freq_dist = nltk.FreqDist(itertools.chain(*output_symbols))
        output_vocab = [y for y, c in freq_dist.most_common(vocab_size)]

        input_symbols=[' '.join([item if item in input_vocab else '<UNK>' for item in seq]) for seq in  input_symbols]
        output_symbols=[' '.join([item if item in output_vocab else '<UNK>' for item in seq]) for seq in  output_symbols]
        return [[input_symbols[k],y] for k,y in enumerate(output_symbols)]


    @staticmethod
    def rep_pair(SR_input, SR_output, pair):
        input_variable = SR_input.rep_sequence(pair[0])
        target_variable = SR_output.rep_sequence(pair[1])
        return (input_variable, target_variable)


    @staticmethod
    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def filterPair(p, MAX_LENGTH=50):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH


    @staticmethod
    def filterPairs(pairs, MAX_LENGTH=15):
        return [pair for pair in pairs if SequenceReader.filterPair(pair,MAX_LENGTH)]

    @staticmethod
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def read_sequence_pairs(sequence_list_1, seq_1_name, sequence_list_2, seq_2_name):

        print("Reading lines...")
        input_lang = SequenceReader(seq_1_name)
        output_lang = SequenceReader(seq_2_name)

        input_lang.add_sequences(sequence_list_1)
        output_lang.add_sequences(sequence_list_2)

        return input_lang, output_lang

    @staticmethod
    def read_data(sequence_pairs, seq_1_name, seq_2_name, max_length=15, filter_vocab=None):
        print("Total number of sequence pairs %s" % len(sequence_pairs))
        sequence_pairs=SequenceReader.filterPairs(sequence_pairs,max_length)
        print("Trimmed to %s sequence pairs" % len(sequence_pairs))
        input_lang = SequenceReader(seq_1_name)
        output_lang = SequenceReader(seq_2_name)

        [input_lang.add_sequence(seqx) for seqx, seqy in sequence_pairs]
        [output_lang.add_sequence(seqy) for seqx, seqy in sequence_pairs]

        if filter_vocab:
            input_lang.filter_vocabulary(filter_vocab)
            output_lang.filter_vocabulary(filter_vocab)

        print("Counting words...")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, sequence_pairs

    @staticmethod
    def split_dataset(x, y, ratio=[0.7, 0.15, 0.15]):
        '''
        split data into train (70%), test (15%) and valid(15%)
        :param x: input list
        :param y: output list
        :param ratio: train, validation, and test ratio
        :return: return tuple( (trainX, trainY), (testX,testY), (validX,validY) )
        I added randomization to the code I got from  https://github.com/suriyadeepan/practical_seq2seq
        '''
        #
        # number of examples
        data_len = len(x)

        lens = [int(data_len * item) for item in ratio]

        idx_list = list(range(0, data_len))

        random.shuffle(idx_list)

        trainX_seq, trainY_seq = [x[k] for k in idx_list[:lens[0]]], [y[k] for k in idx_list[:lens[0]]]
        testX_seq, testY_seq = [x[k] for k in idx_list[lens[0]:lens[0] + lens[1]]], [y[k] for k in idx_list[
                                                                                                   lens[0]:lens[0] +
                                                                                                           lens[1]]]
        validX_seq, validY_seq = [[k] for k in idx_list[-lens[-1]:]], [y[k] for k in idx_list[-lens[-1]:]]

        return (trainX_seq, trainY_seq), (testX_seq, testY_seq), (validX_seq, validY_seq)

    @staticmethod
    def batch_bucket_generator(x, y, batch_size):
        '''
        :param x: input list
        :param y: output list
        :param ratio: train, validation, and test ratio
        :return: return tuple( (trainX, trainY), (testX,testY), (validX,validY) )
        I added randomization to the code I got from  https://github.com/suriyadeepan/practical_seq2seq
        '''

        size_sorted_index = np.argsort([len(item) for item in x])

        c = 0
        batch_idx = 0
        batch_dict = []
        while c < len(x):
            batch_dict.append(dict())
            batch_dict[-1]['id'] = batch_idx
            '''
             randomize the same length sequences
            '''
            idx_list = list(range(c, min(c + batch_size, len(x))))
            if len(idx_list)<batch_size:
                idx_list=list(range(len(x)-batch_size,len(x)))
            random.shuffle(idx_list)
            temp_x = [x[size_sorted_index[i]] for i in idx_list]
            temp_y = [y[size_sorted_index[i]] for i in idx_list]
            batch_dict[-1]['x'] = temp_x
            batch_dict[-1]['y'] = temp_y
            batch_dict[-1]['min_length_x'] = np.min(
                [len(x[size_sorted_index[i]]) for i in idx_list])
            batch_dict[-1]['max_length_x'] = np.max(
                [len(x[size_sorted_index[i]]) for i in idx_list])
            c = c + batch_size
        return batch_dict


if __name__ == "__main__":

    lines = open('/mounts/data/proj/asgari/bible_files/eng2all/par_file/eng_newliving_eng_newworld2013.txt', encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[l.split(' ||| ')[0] , l.split(' ||| ')[0]] for l in lines]

    pairs= SequenceReader.preprocess_pairs(pairs[0:300], 100, preprocess=False)


    #input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in' , 'copy_out', 15)
    #print(input_lang.rep_sequence('he is a great guy ehsan'))
    # output_lang.filter_vocabulary(100)
    # (trainX_seq, trainY_seq), (testX_seq, testY_seq), (validX_seq, validY_seq) = SequenceReader.split_dataset([x for x,y in sequence_pairs], [y for x,y in sequence_pairs], ratio=[0.8, 0, 0.2])
    # for batch in SequenceReader.batch_bucket_generator(trainX_seq, trainY_seq, 100):
    #     print (batch['min_length_x'])
    #     print (batch['max_length_x'])
    #     print ('########################')
    #
    # n=batch['x'][1].split()+['<EOS>']
    # print ([str(w)+' '+n[k] for k, w in enumerate(list(input_lang.rep_sequence(batch['x'][1]).cpu().data.numpy()))])
