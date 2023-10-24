import sys
import re
import pdb
from collections import defaultdict
import heapq

def backtrack(i, cur, prev, kk):
    if i <= 0:
        return []
    _len = len(opt[i][cur][prev])
    _tuple = heapq.nlargest(_len, opt[i][cur][prev])
    return backtrack(i - _tuple[kk][2], prev, _tuple[kk][1], _tuple[kk][3]) + [cur]


KBEST = 10

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

line = "R A PP U T O PP U"
# line = "H E E S U B U KK U R I S A A TCH I S A I E N T I S U T O"
# line = "B I D E O T E E P U"
text = str(line)
text = "<s> " + text + " </s>"
results = re.split(' |\n', text)
Jphoneme = list(filter(None, results))

opt = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

opt[0]['<s>']['<s>'].append((1, '<s>', 0, 0))
heapq.heapify(opt[0]['<s>']['<s>'])

for i in range(1, len(Jphoneme)):
    for j in tag_set:
        for k in tag_set:
            heapq.heapify(opt[i][j][k])

for i in range(1, len(Jphoneme)):
    for k in range(1, 4):

        if i - k + 1 < 0:
            continue
        Ji = ""
        for l in range(i - k + 1, i + 1):
            Ji += Jphoneme[l]
        if Ji not in pwords:
            continue

        for e in pwords[Ji]:
            for eprev in peprob[e]:
                for eprevprev in peprob[e][eprev]:

                        prev_best = heapq.nlargest(min(len(opt[i-k][eprev][eprevprev]), KBEST), opt[i-k][eprev][eprevprev])
                        if len(prev_best) == 0:
                            continue
                        for zz in range(len(prev_best)):

                            score = float(prev_best[zz][0]) * peprob[e][eprev][eprevprev] * pwords[Ji][e]
                            opt[i][e][eprev].append((score, eprevprev, k, zz))

kbest_list = []
_len = len(Jphoneme) - 1

tag_set1 = ['</s>']
for i in tag_set1:
    for j in tag_set:

        if len(opt[_len][i][j]) == 0:
            continue
        _tuple = heapq.nlargest(len(opt[_len][i][j])-1, opt[_len][i][j])
        for z in range(len(_tuple)):
            kbest_list.append((_tuple[z][0], z, i, j))

heapq.heapify(kbest_list)
kbest_list = heapq.nlargest(KBEST, kbest_list)

for kbest in range(len(kbest_list)):
    ret = ' '.join(backtrack(_len, kbest_list[kbest][2], kbest_list[kbest][3], kbest_list[kbest][1])[:-1])
    print(str(ret) + " # " + str(kbest_list[kbest][0]))
