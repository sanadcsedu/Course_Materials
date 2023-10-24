import sys
import re
import pdb
from collections import defaultdict
import heapq
import time


def backtrack(i, cur, prev):
    if i <= 0:
        return []
    return backtrack(i - bestk[i][cur][prev], prev, beste[i][cur][prev]) + [cur]


peprob = {}
pwords = {}

try:
    epron_prob_file, epron_jpron_prob_file = sys.argv[1:]

    peprob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with open(epron_prob_file, "r") as f:
        for line in f:
            text = str(line)
            results = re.split(' |:|#|\n', text)
            results = list(filter(None, results))
            peprob[results[2]][results[1]][results[0]] = float(results[3])

    pwords = defaultdict(lambda: defaultdict(float))
    with open(epron_jpron_prob_file, "r") as f:
        for line in f:
            text = str(line)
            results = re.split(' |:|#|\n', text)
            results = list(filter(None, results))
            temp = ""
            for i in range(1, len(results) - 1):
                temp += results[i]
            pwords[temp][results[0]] = float(results[len(results) - 1])
    pwords['</s>'] = {}
    pwords['</s>']['</s>'] = 1

except:
    sys.stderr.write("usage: decode.py <epron_prob_file> <epron_jpron_prob_file>\n")
    sys.exit(1)

for linenum, line in enumerate(sys.stdin):
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
            for l in range(i - k + 1, i+1):
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
    _list = []
    _len = len(Jphoneme) - 1
    for j in opt[_len]['</s>']:
        _list.append((opt[_len]['</s>'][j], '</s>', j))
    heapq.heapify(_list)

    kbest = heapq.nlargest(1, _list)

    ret = ' '.join(backtrack(_len, kbest[0][1], kbest[0][2])[:-1])
    print(str(ret) + " # " + str(kbest[0][0]))
print(time.clock() - start_time)