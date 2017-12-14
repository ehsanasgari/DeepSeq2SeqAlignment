# Author: Ehsaneddin Asgari
# Email: asgari@berkeley.edu
# Evaluation against the gold standard


import itertools
import codecs
import os
import fnmatch
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

class Evaluation(object):
    def __init__(self, dir_path):
        self.gold_standard_instances=dict()
        self.baseline_instances=dict()
        self.read_gold_standard(Evaluation.recursive_glob(dir_path+'/goldstd/','*.wa'))
        self.read_baseline(Evaluation.recursive_glob(dir_path+'/baseline/','*.align'))
    
    def read_gold_standard(self,files):
        for file in files:
            l1,l2=file.split('/')[-1].split('_')[0].split('-')
            for line in open(file, 'r').readlines():
                toks=line.strip().split()
                toks[0]=int(toks[0])
                if toks[0] not in self.gold_standard_instances:
                    self.gold_standard_instances[toks[0]]=dict()
                if (l1,l2) not in self.gold_standard_instances[toks[0]]:
                    self.gold_standard_instances[toks[0]][(l1,l2)]=[]
                if (l2,l1) not in self.gold_standard_instances[toks[0]]:
                    self.gold_standard_instances[toks[0]][(l2,l1)]=[]
                self.gold_standard_instances[toks[0]][(l1,l2)].append(((int(toks[1])-1,int(toks[2])-1),toks[3]))
                self.gold_standard_instances[toks[0]][(l2,l1)].append(((int(toks[2])-1,int(toks[1])-1),toks[3]))
    
    def read_baseline(self, files):
        for file in files:
            l1,l2=file.split('/')[-1].split('.')[0].split('_')
            for idx, line in enumerate(open(file, 'r').readlines()):
                line_num=idx+1
                algn=[tuple([int(x) for x in algn.split('-')]) for algn in line.strip().split()]
                if line_num not in self.baseline_instances:
                    self.baseline_instances[line_num]=dict()
                self.baseline_instances[line_num][(l1,l2)]=algn
    
    def evaluate_P_R_AER(self, alignment_instances):
        A=0; AS=0; AP=0; S=0
        for k in range(1,101):
            for langs in alignment_instances[k].keys():
                dA,dAS,dAP,dS=Evaluation._return_ARE_res(alignment_instances[k][langs],self.gold_standard_instances[k][langs])
                A+=dA
                AS+=dAS
                AP+=dAP
                S+=dS
        return ('Precision', AP/A) , ('Recall',AS/S), ('AER', 1 - (AS + AP) / (A + S))
    
    @staticmethod
    def _return_ARE_res(obs,gt):
        A=0; AS=0; AP=0; S=0
        check_dict=dict(gt)
        for x in obs:
            if x in check_dict:
                if check_dict[x]=='S':
                    AS+=1
                    AP+=1
                elif check_dict[x]=='P':
                    AP+=1
        S=list(check_dict.values()).count('S')
        return len(obs), AS,AP, S
        
                
                
    @staticmethod
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
        