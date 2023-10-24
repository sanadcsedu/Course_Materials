import sys
import re
import pdb
from collections import defaultdict
import heapq

def checkkey(_dict, u, v, w):
    if u in _dict:
        if v in _dict[u]:
            if w in _dict[u][v]:
                return True
    return False


def checkkey2(_dict, u, v):
    if u in _dict:
        if v in _dict[u]:
            return True
    return False


def backtrack(i, cur, prev):
    if i <= 0:
        return []
    #print("--> {} {} {} {} {}", i, cur, prev, beste[i][cur][prev], bestk[i][cur][prev], opt[i][cur][prev])
    return backtrack(i - bestk[i][cur][prev], prev, beste[i][cur][prev]) + [cur]



f = open("epron.probs", "r")
peprob = {}
tag_set = set()
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
f.close()

f = open("epron-jpron.probs", "r")
pwords = {}
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

line = "T O R I P U R U R U U M U"
text = str(line)
text = "<s> " + text + " </s>"
results = re.split(' |\n', text)
Jphoneme = list(filter(None, results))

opt = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
bestk = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
beste = defaultdict(lambda : defaultdict(lambda : defaultdict()))

opt[0]['<s>']['<s>'] = 1

for i in range(1, len(Jphoneme)):
    for e in tag_set:
        for eprev in tag_set:
            for eprevprev in tag_set:
                for k in range(1, 4):
                    if i - k + 1 < 0:
                        continue
                    Ji = ""
                    for l in range (i - k + 1, i+1):
                        Ji += Jphoneme[l]

                    if checkkey(peprob, e, eprev, eprevprev) == False:
                        continue

                    if (Ji not in pwords) or (e not in pwords[Ji]):
                        continue

                    score = opt[i-k][eprev][eprevprev] * peprob[e][eprev][eprevprev] * pwords[Ji][e]
                    if score > opt[i][e][eprev]:
                        opt[i][e][eprev] = score
                        beste[i][e][eprev] = eprevprev
                        bestk[i][e][eprev] = k

maxe = -1
_len = len(Jphoneme) - 1
for i in tag_set:
    for j in tag_set:
        if maxe < opt[_len][i][j]:
               maxe = opt[_len][i][j]
               res_e = i
               res_eprev = j
ret = ' '.join(backtrack(_len, res_e, res_eprev)[:-1])
print(str(ret) + " # " + str(maxe))
