import torch
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

sys.stdout = open("test.txt.novowels_new.base", 'w')

with open("test.txt.novowels") as fp:
    for lines in fp:
        lines = lines.strip()

        h = nlm.NLM()
        beam = [(0, h)]
        len_lines = len(lines)
        for ii in tqdm(range(len_lines)):
            c = lines[ii]
            if c == ' ':
                c = '_'

            temp = []
            for (p, h) in beam:

                prob = p + math.log(h.next_prob(c))
                h_temp = h + c
                temp.append((prob, h_temp))

                for permutes in perm:
                    h_temp = h + ""
                    prob = p
                    for j in permutes:
                        prob += math.log(h_temp.next_prob(j))
                        h_temp += j

                    prob += math.log(h_temp.next_prob(c))
                    h_temp += c
                    temp.append((prob, h_temp))

            beam = sorted(temp, reverse=True)[:beam_len]

        p, h = beam[0]
        print(("".join(h.history)).replace("_", " ").replace("<s>", ""))
