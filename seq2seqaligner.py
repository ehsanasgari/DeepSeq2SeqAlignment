import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import random
import codecs
from seq2seq_reader import SequenceReader
from models import EncoderRNN, AttnDecoderRNN
from utility import as_minutes, time_since, show_plot
import numpy as np
import pickle
import itertools

class Seq2SeqAligner(object):
    '''
        This class is written using pytorch for an alignment method using encoder-decoder architecture
    '''
    use_cuda = torch.cuda.is_available()
    def __init__(self, SR_input, SR_output, sequence_pairs, hidden_size = 256, embedding_size = 128, n_decode_layer=1, max_length=50, bidirectional=False,enc_dropout=0.1,att_dropout=0.1):

        if bidirectional:
            self.bi_coef = 2
        else:
            self.bi_coef = 1
        self.max_length = max_length
        self.encoder = EncoderRNN(SR_input.n_words, embedding_size, hidden_size, dropout=enc_dropout,
                                  bidirectional=bidirectional)
        self.attn_decoder = AttnDecoderRNN(hidden_size * self.bi_coef, SR_output.n_words, n_decode_layer, dropout_p=att_dropout,
                                           max_length_out=max_length)

        if Seq2SeqAligner.use_cuda:
            print ('CUDA in use...')
        else:
            print ('CPU in use...')
        if Seq2SeqAligner.use_cuda:
            self.encoder = self.encoder.cuda()
            self.attn_decoder = self.attn_decoder.cuda()

        self.sequence_pairs = sequence_pairs
        self.SR_input = SR_input
        self.SR_ouput = SR_output
        (self.trainX_seq, self.trainY_seq), (self.testX_seq, self.testY_seq), (
            self.validX_seq, self.validY_seq) = SequenceReader.split_dataset([x for x, y in sequence_pairs],
                                                                             [y for x, y in sequence_pairs],
                                                                             ratio=[0.8, 0.2, 0])

    def train_iteration_bucket(self, n_epoch, batchsize=50, print_every=100, validation_every=1000, learning_rate=0.01):
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.attn_decoder.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()
        iter = 0
        for epoch in range(n_epoch):
            for batch in SequenceReader.batch_bucket_generator(self.trainX_seq, self.trainY_seq, batchsize):
                for i in range(batchsize):
                    input_variable = self.SR_input.rep_sequence(batch['x'][i])
                    target_variable = self.SR_ouput.rep_sequence(batch['y'][i])

                    loss = self.train(input_variable, target_variable, criterion, 0.5)
                    print_loss_total += loss

                    if iter % print_every == 0:
                        print_loss_avg = print_loss_total / print_every
                        print_loss_total = 0
                        print('average train loss %.4f' % print_loss_avg)

                    if iter % validation_every == 0:
                        validation_loss = 0
                        count = 0
                        for val_batch in SequenceReader.batch_bucket_generator(self.testX_seq, self.testY_seq,
                                                                               batchsize):
                            for j in range(batchsize):
                                output_words, attentions, l, c = self.evaluate(val_batch['x'][j], val_batch['y'][j])
                                validation_loss += l
                                count = count + c
                        l = validation_loss / count
                        print('average validation loss %.4f' % l)
                        idx = random.randint(0, len(self.testX_seq))
                        print('>', self.testX_seq[idx])
                        print('=', self.testY_seq[idx])
                        output_words, attentions, l, c = self.evaluate(self.testX_seq[idx], self.testY_seq[idx])
                        output_sentence = ' '.join(output_words)
                        print('<', output_sentence)
                        print('')

                    iter += 1

    def train_iteration(self, n_epoch, print_every=100, validation_every=1000, learning_rate=0.01,name='', teaching=0.5):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        validation_loss = 0
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.attn_decoder.parameters(), lr=learning_rate)

        f=codecs.open(name+'.log','w','utf-8')
        for epoch in range(1, n_epoch + 1):
            pairs = [[self.trainX_seq[idx], self.trainY_seq[idx]] for idx in range(len(self.trainX_seq))]
            x, y = random.choice(pairs)
            # Get training data for this cycle
            input_variable = self.SR_input.rep_sequence(x)
            target_variable = self.SR_ouput.rep_sequence(y)

            criterion = nn.NLLLoss()
            # Run the train function
            loss = self.train(input_variable, target_variable, criterion, teaching)
            # Keep track of loss
            print_loss_total += loss

            if epoch == 0: continue

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, epoch / n_epoch), epoch, epoch / n_epoch * 100, print_loss_avg)
                print(print_summary)
                f.write(print_summary+'\n')

            if epoch % validation_every == 0:
                print (epoch)
                validation_loss = 0
                for val_batch in SequenceReader.batch_bucket_generator(self.testX_seq, self.testY_seq, 1):
                    output_words, attentions, l = self.evaluate(val_batch['x'][0], val_batch['y'][0])
                    validation_loss += l
                l = validation_loss / len(self.testX_seq)
                print('average validation loss %.4f' % l)
                f.write(('average validation loss %.4f' % l)+'\n')


                for i in range(1,10):
                    idx = random.randint(0, len(self.testX_seq)-1)
                    print('>', self.testX_seq[idx])
                    print('=', self.testY_seq[idx])
                    output_words, attentions, l = self.evaluate(self.testX_seq[idx], self.testY_seq[idx])
                    output_sentence = ' '.join(output_words)
                    print('<', output_sentence)
                    print('')

        for val_batch in SequenceReader.batch_bucket_generator(self.testX_seq, self.testY_seq, 1):
            output_words, attentions, l = self.evaluate(val_batch['x'][0], val_batch['y'][0])
            validation_loss += l
            f.write('> '+ val_batch['x'][0]+'\n')
            f.write('= '+ val_batch['y'][0]+'\n')
            f.write('< '+ ' '.join(output_words)+'\n\n')
        f.close()

    def train(self, input_variable, target_variable, criterion, teacher_forcing_ratio):

        encoder_hidden = self.encoder.initHidden()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size * self.bi_coef))
        encoder_outputs = encoder_outputs.cuda() if Seq2SeqAligner.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SequenceReader.GO_token]]))
        decoder_input = decoder_input.cuda() if Seq2SeqAligner.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(decoder_input, decoder_hidden,
                                                                                      encoder_output, encoder_outputs)
                loss += criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(decoder_input, decoder_hidden,
                                                                                      encoder_output, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if Seq2SeqAligner.use_cuda else decoder_input

                loss += criterion(decoder_output[0], target_variable[di])
                if ni == SequenceReader.EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / target_length

    def evaluate(self, sequence, target):
        input_variable = self.SR_input.rep_sequence(sequence)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size * self.bi_coef))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                          encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SequenceReader.GO_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        loss = 0
        out_length=0
        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            #print(decoder_output[0])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == SequenceReader.EOS_token:
                decoded_words.append('<EOS>')
                out_length=di+1
                break
            else:
                decoded_words.append(self.SR_ouput.index2word[ni])
                if di >= len(target):
                    loss+=1
                else:
                    loss += 0 if (decoded_words[-1]== target[di]) else 1

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        if out_length==0:
            out_length=self.max_length

        return decoded_words, decoder_attentions[:di + 1], loss / out_length

    def evaluate_attention(self, sequence):
        input_variable = self.SR_input.rep_sequence(sequence)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size * self.bi_coef))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                          encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SequenceReader.GO_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        loss = 0
        out_length=0
        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            #print(decoder_output[0])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == SequenceReader.EOS_token:
                decoded_words.append('<EOS>')
                out_length=di+1
                break
            else:
                decoded_words.append(self.SR_ouput.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        if out_length==0:
            out_length=self.max_length

        return decoded_words, decoder_attentions[:di + 1]

def sample_run():
    lines = open('/mounts/data/proj/asgari/bible_files/eng2all/par_file/eng_newliving_eng_newworld2013.txt',
                 encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[l.split(' ||| ')[0], l.split(' ||| ')[0]] for l in lines]
    pairs = SequenceReader.preprocess_pairs(pairs, 100, preprocess=False)
    # lines = open('uniqdataset.txt',
    #             encoding='utf-8'). \
    #    read().strip().split('\n')
    # pairs = [[l.split(' * ')[0], l.split(' * ')[0]] for l in lines]

    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', 15)

    (trainX_seq, trainY_seq), (testX_seq, testY_seq), (validX_seq, validY_seq) = SequenceReader.split_dataset(
        [x for x, y in sequence_pairs], [y for x, y in sequence_pairs], ratio=[0.8, 0, 0.2])

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, n_decode_layer=1, bidirectional=True)

    SSA.train_iteration(300, 50)


def toy_hidden_size():



    max_length=10
    vocab_size=30
    hidden_sizes=[128, 256, 512]
    embedding_sizes=[100,200,400]
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1

    for hidden_size in hidden_sizes:
        for embedding_size in embedding_sizes:
            for i in range(20000):
                len = random.randint(1, max_length)
                seq = ' '.join([str(x) for x in random.sample(range(vocab_size), len)])
                pairs.append([seq, seq])

            name= '_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout)])
            input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

            SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
            SSA.train_iteration(70000, print_every, validate_every,learning_rate, name)

def toy_layers():
    max_length=10
    vocab_size=30
    hidden_sizes=[128, 256]
    embedding_sizes=[200,400]
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layerss=[2,3,4]
    enc_dropout=0.1
    att_dropout=0.1

    for enc_layers in enc_layerss:
        for hidden_size in hidden_sizes:
            for embedding_size in embedding_sizes:
                for i in range(20000):
                    len = random.randint(1, max_length)
                    seq = ' '.join([str(x) for x in random.sample(range(vocab_size), len)])
                    pairs.append([seq, seq])

                name= '_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout)])
                input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

                SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
                SSA.train_iteration(70000, print_every, validate_every,learning_rate, name)

def toy_dropout():
    max_length=10
    vocab_size=30
    hidden_sizes=[128, 256]
    embedding_sizes=[200,400]
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=1
    enc_dropouts=[0.2,0.3]
    att_dropouts=[0.2,0.3]

    for enc_dropout in enc_dropouts:
        for att_dropout in att_dropouts:
            for hidden_size in hidden_sizes:
                for embedding_size in embedding_sizes:
                    for i in range(20000):
                        len = random.randint(1, max_length)
                        seq = ' '.join([str(x) for x in random.sample(range(vocab_size), len)])
                        pairs.append([seq, seq])

                    name= '_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout)])
                    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

                    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
                    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name)

def toy_teaching():
    max_length=10
    vocab_size=30
    hidden_sizes=[128, 256]
    embedding_sizes=[200,400]
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layerss=[1,2]
    enc_dropout=0.1
    att_dropout=0.1
    teachings=[0.1,0.3,0.5,0.7,1]

    for teaching in teachings:
        for enc_layers in enc_layerss:
            for hidden_size in hidden_sizes:
                for embedding_size in embedding_sizes:
                    for i in range(20000):
                        len = random.randint(1, max_length)
                        seq = ' '.join([str(x) for x in random.sample(range(vocab_size), len)])
                        pairs.append([seq, seq])

                    name= '_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
                    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

                    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
                    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def toy_size():
    max_length=20
    vocab_size=20
    hidden_sizes=[128, 256]
    embedding_sizes=[200,400]
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layerss=[1,2]
    enc_dropout=0.1
    att_dropout=0.1
    teachings=[0.3,0.5,0.7]

    for teaching in teachings:
        for enc_layers in enc_layerss:
            for hidden_size in hidden_sizes:
                for embedding_size in embedding_sizes:
                    for i in range(20000):
                        len = random.randint(1, max_length)
                        seq = ' '.join([str(x) for x in random.sample(range(vocab_size), len)])
                        pairs.append([seq, seq])

                    name= '_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
                    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

                    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
                    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def toy_enfr():
    lines = open('uniqdataset.txt',
                 encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[l.split(' * ')[0], l.split(' * ')[0]] for l in lines]
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', 15)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, n_decode_layer=1, bidirectional=True)
    SSA.train_iteration(75000, 5000)

def best_models():
    max_length=10
    vocab_size=30
    hidden_size=256
    embedding_size=200
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=2
    enc_dropout=0.1
    att_dropout=0.1
    teaching=0.5

    for i in range(20000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_models2():
    max_length=10
    vocab_size=30
    hidden_size=256
    embedding_size=200
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=2
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1

    for i in range(20000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()

    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_models_40():
    max_length=15
    vocab_size=1000
    hidden_size=512
    embedding_size=1000
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=2
    enc_dropout=0.1
    att_dropout=0.1
    teaching=0.5

    for i in range(20000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model40_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()

    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_models_40_2():
    max_length=10
    vocab_size=40
    hidden_size=256
    embedding_size=200
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=2
    enc_dropout=0.1
    att_dropout=0.1
    teaching=0.5

    for i in range(20000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model40_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_50():
    max_length=10
    vocab_size=50
    hidden_size=128
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=50000

    for i in range(50000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_100():
    max_length=10
    vocab_size=100
    hidden_size=128
    embedding_size=200
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1

    for i in range(20000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_100():
    max_length=10
    vocab_size=50
    hidden_size=128
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=50000
    for i in range(training_size):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_1000():
    max_length=10
    vocab_size=1000
    hidden_size=256
    embedding_size=1024
    print_every=5000
    validate_every=10000
    learning_rate=0.01
    pairs = []
    enc_layers=3
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1

    for i in range(100000):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)


def best_110():
    '''
    hidden 256-100-
    128-100-2
    :return:
    '''
    max_length=10
    vocab_size=110
    hidden_size=256
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=50000

    for i in range(training_size):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_500():
    '''
    hidden 256-100-
    128-100-2
    :return:
    '''
    max_length=10
    vocab_size=500
    hidden_size=256
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=100000

    for i in range(training_size):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(200000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_80():
    '''
    hidden 256-100-
    128-100-2
    :return:
    '''
    max_length=10
    vocab_size=80
    hidden_size=128
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=2
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=50000

    for i in range(training_size):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(70000, print_every, validate_every,learning_rate, name, teaching=teaching)

def best_1000():
    '''
    hidden 256-100-
    128-100-2
    :return:
    '''
    max_length=10
    vocab_size=1000
    hidden_size=350
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=1
    training_size=100000
    iteration=350000
    for i in range(training_size):
        len = random.randint(1, max_length)
        seq = ' '.join([str(x) for x in np.random.choice(range(vocab_size), len)])
        pairs.append([seq, seq])

    name= 'best_model_'+'_'.join(['max_length',str(max_length),'vocab_size',str(vocab_size),'hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'training_size',str(training_size),'iterations',str(iteration)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'copy_in', 'copy_out', max_length)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(iteration, print_every, validate_every,learning_rate, name, teaching=teaching)


    with open(name+'.pkl', 'wb') as input:
        pickle.dump(SSA,input)

def best_eng2eng():
    '''
    hidden 256-100-
    128-100-2
    :return:
    '''
    hidden_size=350
    embedding_size=100
    print_every=5000
    validate_every=10000
    learning_rate=0.005
    pairs = []
    enc_layers=1
    enc_dropout=0.1
    att_dropout=0.1
    teaching=0.5
    iteration=350000#150000#

    first=[line.strip() for line in codecs.open('first.txt','r','utf-8').readlines()]
    second=[line.strip() for line in codecs.open('second.txt','r','utf-8').readlines()]

    pairs_first =[pairs[0] +' '+ pairs[1] for pairs in list(itertools.product(first, first))]
    pairs_second =[pairs[0]+' '+pairs[1] for pairs in list(itertools.product(second, second))]

    pairs=[]
    for k,pair in enumerate(pairs_first):
        pairs.append([pairs_first[k],pairs_second[k]])

    name= 'best_model_eng_eng_l20'+'_'.join(['hidden_size',str(hidden_size),'embedding_size',str(embedding_size),'enc_layers',str(enc_layers),'enc_dropout',str(enc_dropout),'att_dropout',str(att_dropout),'teaching', str(teaching),'iterations',str(iteration)])
    f=codecs.open(name+'_corpus.txt','w','utf-8')
    for x in pairs:
        f.write(x[0]+'\n')
    f.close()
    input_lang, output_lang, sequence_pairs = SequenceReader.read_data(pairs, 'newliving', 'treeoflife', 20)

    SSA = Seq2SeqAligner(input_lang, output_lang, sequence_pairs, hidden_size, embedding_size, n_decode_layer=enc_layers, bidirectional=True, enc_dropout=enc_dropout,att_dropout=att_dropout)
    SSA.train_iteration(iteration, print_every, validate_every,learning_rate, name, teaching=teaching)


    with open(name+'.pkl', 'wb') as input:
        pickle.dump(SSA,input)

if __name__ == "__main__":
    best_eng2eng()
