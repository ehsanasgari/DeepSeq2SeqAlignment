import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import codecs

import os
import fnmatch

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def recursive_glob(treeroot, pattern):
    '''
    :param treeroot: the path to the directory
    :param pattern:  the pattern of files
    :return:
    '''
    results = []
    for base, dirs, files in os.walk(treeroot):
        good_files = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in good_files)
    return results

def explore_parameters():
    files=recursive_glob('./','max_length*')
    print('1) max_length	2) vocab_size	3) hidden_size	4) embedding_size	5) number_layers	6) enc_do	7) att_do	8) teaching	train_loss	valid_accuracy')
    for file in files:
        lines=codecs.open(file,'r','utf-8').readlines()
        if len(lines)==21:
            a=[x.split()[-1] for x in lines[-2::]]
            param=file.split('_')
            params=dict()
            params['1) max_length']=param[2]
            params['2) vocab_size']=param[5]
            params['3) hidden_size']=param[8]
            params['4) embedding_size']=param[11]
            params['5) number_layers']=param[14]
            params['6) enc_do']=param[17]
            if 'teaching' not in param:
                params['7) att_do']=param[20].replace('.log','')
                params['8) teaching']='0.5'
            else:
                params['7) att_do']=param[20]
                params['8) teaching']=param[22].replace('.log','')

            pr=list(params.keys())
            pr.sort()
            pr.append('train_loss')
            pr.append('valid_accuracy')
            params['train_loss']=a[0]
            params['valid_accuracy']=a[1]

            print('\t'.join([params[p] for p in pr]))


if __name__ == "__main__":
    explore_parameters()
