import torch
import nlm
import sys
import math
from tqdm import tqdm
from pprint import pprint
from itertools import product
import pdb
import math
from collections import defaultdict
nlm.NLM.load('large')
from itertools import permutations
beam_len = 40
vowel = ['a', 'e', 'i', 'o', 'u']
perm = []

for i in vowel:
    perm.append([i])
for i in vowel:
    for j in vowel:
        perm.append([i, j])
for i in vowel:
    for j in vowel:
        for k in vowel:
            perm.append([i, j, k])
cnt = 0
# sys.stdout = open("test.txt.novowels.large", 'w')
ans = []
with open("test.txt") as fp:
    cnt += 1
    for lines in fp:
        lines = lines.strip()
        tempv = []
        for c in lines:
            if c in vowel:
                tempv += c
            else:
                tempv = []
            if len(tempv) >= 3:
                ans.append(tempv)
print(ans)
        # print(lines)

        # h = nlm.NLM()
        # beam = [(0, h)]
        # len_lines = len(lines)
        # for ii in tqdm(range(len_lines)):
        #     c = lines[ii]
        #     if c == ' ':
        #         c = '_'
        #
        #     temp = []
        #     for (p, h) in beam:
        #
        #         prob = p + math.log(h.next_prob(c))
        #         h_temp = h + c
        #         temp.append((prob, h_temp))
        #
        #         for permutes in perm:
        #             h_temp = h + ""
        #             prob = p
        #             for j in permutes:
        #                 prob += math.log(h_temp.next_prob(j))
        #                 h_temp += j
        #
        #             prob += math.log(h_temp.next_prob(c))
        #             h_temp += c
        #             temp.append((prob, h_temp))
        #
        #     beam = sorted(temp, reverse=True)[:beam_len]
        #
        # p, h = beam[0]
        # print(("".join(h.history)).replace("_", " ").replace("<s>", ""))
        # if cnt == 3:
        #     break
