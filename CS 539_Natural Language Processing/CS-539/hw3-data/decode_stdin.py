import sys
import re
import pdb
from collections import defaultdict
import heapq

def backtrack(i, cur, prev):
    if i <= 0:
        return []
    return backtrack(i - bestk[i][cur][prev], prev, beste[i][cur][prev]) + [cur]

tag_set = set()
peprob = {}
pwords = {}
try:
    epron_prob_file, epron_jpron_prob_file = sys.argv[1:]

    with open(epron_prob_file, 'r') as f:
        for line in f:
            text = str(line)
            results = re.split(' |:|#|\n', text)
            results = list(filter(None, results))
            if results[2] not in peprob:
                peprob[results[2]] = {}
            if results[1] not in peprob[results[2]]:
                peprob[results[2]][results[1]] = {}
            if results[0] not in peprob[results[2]][results[1]]:
                peprob[results[2]][results[1]][results[0]] = 0
            peprob[results[2]][results[1]][results[0]] = float(results[3])
            tag_set.add(results[2])
            tag_set.add(results[1])
            tag_set.add(results[0])

    with open(epron_jpron_prob_file, 'r') as f:
        for line in f:
            text = str(line)
            results = re.split(' |:|#|\n', text)
            results = list(filter(None, results))
            temp= ""
            for i in range(1, len(results) - 1):
                temp += results[i]
            if temp not in pwords:
                pwords[temp] = {}
            if results[0] not in pwords[temp]:
                pwords[temp][results[0]] = 0
            pwords[temp][results[0]] = float(results[len(results) - 1])
    pwords['</s>'] = {}
    pwords['</s>']['</s>'] = 1
except:
    sys.stderr.write("usage: decode.py <epron_prob_file> <epron_jpron_prob_file>\n")
    sys.exit(1)

for linenum, line in enumerate(['P I A N O']):#sys.stdin):
    text = str(line.rstrip())
    text = "<s> " + text + " </s>"
    results = re.split(' |\n', text)
    Jphoneme = list(filter(None, results))

    opt = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    bestk = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    beste = defaultdict(lambda : defaultdict(lambda : defaultdict()))

    opt[0]['<s>']['<s>'] = 1

    for i in range(1, len(Jphoneme)):
        for k in range(1, 4):
        
            if i - k + 1 < 0:
                continue
            Ji = ""
            for l in range (i - k + 1, i+1):
                Ji += Jphoneme[l]
            
            if Ji not in pwords:
                continue
            
            for e in pwords[Ji]:
                for eprev in peprob[e]:
                    for eprevprev in peprob[e][eprev]:
                            score = opt[i-k][eprev][eprevprev] * peprob[e][eprev][eprevprev] * pwords[Ji][e]
                            if score > opt[i][e][eprev]:
                                opt[i][e][eprev] = score
                                beste[i][e][eprev] = eprevprev
                                bestk[i][e][eprev] = k

    K = 1
    _list = []
    _len = len(Jphoneme) - 1
    for i in tag_set:
        for j in tag_set:
            if opt[_len][i][j] > 0:
                _list.append((opt[_len][i][j], i, j))
    heapq.heapify(_list)

    kbest = heapq.nlargest(K, (_list))
    for i in range(len(kbest)):
        res_e = kbest[i][1]
        res_eprev = kbest[i][2]
        maxe = kbest[i][0]
    
    ret = ' '.join(backtrack(_len, kbest[i][1], res_eprev)[:-1])
    print(str(ret) + " # " + str(maxe))
