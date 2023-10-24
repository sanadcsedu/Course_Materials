#import torch
import nlm
import sys
import math
from tqdm import tqdm
from pprint import pprint
from itertools import product
import pdb
import math
nlm.NLM.load('base')
from itertools import permutations
beam_len = 40
vowel = ['a', 'e', 'i', 'o', 'u']
perm = []

for i in vowel:
    perm.append([i])
for i in vowel:
    for j in vowel:
        perm.append([i, j])
# for i in vowel:
#     for j in vowel:
#         for k in vowel:
#             perm.append([i, j, k])
# lines = "yt dmnstrtn ffcls hv bgn t dscrb clmb s nthr grv strtgc rsk"
for lines in sys.stdin.readlines():
    lines = lines.strip()
    h = nlm.NLM()
    beam = []
    beam.append((0, h))
    len_lines = len(lines)
    for ii in tqdm(range(len_lines)):
        c = lines[ii]
        if c == ' ':
            c = '_'
            h = nlm.NLM()

        temp = []
        for (p, h) in beam:

            tprob = p + math.log(h.next_prob(c))
            th_temp = h + c
            temp.append((tprob, th_temp))

            for permutes in perm:
                h_temp = h + ""
                prob = p
                cnt = 0
                for j in permutes:
                    prob += math.log(h_temp.next_prob(j))
                    if prob < tprob:
                        break
                    h_temp += j
                    cnt += 1

                if cnt == len(permutes):
                    # print("here")
                    prob = p + math.log(h_temp.next_prob(c))
                    h_temp += c
                    temp.append((prob, h_temp))
                    # print(len(temp))

                # pdb.set_trace()
                # if prob > tprob and c == 't':
                #     pdb.set_trace()

        beam = sorted(temp, reverse=True)[:beam_len]

    p, h = beam[0]
    print(("".join(h.history)).replace("_", " ").replace("<s>", ""))