import sys
import re
import pdb
from collections import defaultdict
import heapq
import time

def backtrack(i, cur, prev, kk):
    if i <= 0:
        return []
    top_tuples = heapq.nlargest(kk+1, opt[i][cur][prev])
    return backtrack(i - top_tuples[kk][2], prev, top_tuples[kk][1], top_tuples[kk][3]) + [cur]


peprob = {}
tag_set = set()
pwords = {}

try:
    epron_prob_file, epron_jpron_prob_file, k_best = sys.argv[1:]
    k_best = int(k_best)

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
    sys.stderr.write("usage: kbest.py <epron_prob_file> <epron_jpron_prob_file> <k_best>\n")
    sys.exit(1)

for linenum, line in enumerate(sys.stdin):
    text = str(line.rstrip())
    text = "<s> " + text + " </s>"
    results = re.split(' |\n', text)
    Jphoneme = list(filter(None, results))

    opt = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    opt[0]['<s>']['<s>'].append((1, '<s>', 0, 0))
    heapq.heapify(opt[0]['<s>']['<s>'])

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
                    for eprevprev in opt[i-k][eprev]:

                            prev_best = heapq.nlargest(min(len(opt[i-k][eprev][eprevprev]), k_best), opt[i-k][eprev][eprevprev])
                            if len(prev_best) == 0:
                                continue
                            for zz in range(len(prev_best)):

                                score = float(prev_best[zz][0]) * peprob[e][eprev][eprevprev] * pwords[Ji][e]
                                opt[i][e][eprev].append((score, eprevprev, k, zz))
                    heapq.heapify(opt[i][e][eprev])

    kbest_list = []
    input_len = len(Jphoneme) - 1

    for j in opt[input_len]['</s>']:
        top_tuples = heapq.nlargest(len(opt[input_len]['</s>'][j])-1, opt[input_len]['</s>'][j])
        for z in range(len(top_tuples)):
            kbest_list.append((top_tuples[z][0], z, '</s>', j))

    heapq.heapify(kbest_list)
    kbest_list = heapq.nlargest(k_best, kbest_list)

    for kbest in range(len(kbest_list)):
        ret = ' '.join(backtrack(input_len, kbest_list[kbest][2], kbest_list[kbest][3], kbest_list[kbest][1])[:-1])
        print(str(ret) + " # " + str(kbest_list[kbest][0]))
