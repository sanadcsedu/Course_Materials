#!/usr/bin/env python

import pdb
import sys, itertools
from collections import defaultdict

tags = {'I': ['PRO'], 'HOPE': ['V'], 'THAT': ['PRO', 'CONJ'], 'THIS': ['PRO', 'DT'],
    'WORKS': ['V'], 'THEY': ['PRO'], 'CAN': ['V', 'N', 'AUX'], 'FISH': ['V', 'N'],
    'A': ['DT'], 'PANDA': ['N'], 'EATS': ['V'], 'SHOOTS': ['V', 'N'], 'AND': ['CONJ'],
    'LEAVES': ['V', 'N'], 'LEAVES': ['V'], 'TIME': ['N'], 'FLIES': ['V', 'N'],
    'LIKE': ['PREP'], 'AN': ['DT'], 'ARROW': ['N'], '</s>': ['</s>']}
ptag = {'<s>': {'PRO': 0.6, 'N': 0.2, 'DT': 0.2},
    'PRO': {'V': 0.5, 'AUX': 0.3, 'CONJ': 0.2},
    'N': {'V': 0.3, 'AUX': 0.2, '</s>': 0.5},
    'DT': {'N': 1.0},
    'AUX': {'V': 1.0},
    'V': {'V': 0.1, 'DT': 0.2, 'PRO': 0.2, 'N': 0.3, 'CONJ': 0.05, 'PREP': 0.05, '</s>': 0.1},
    'PREP': {'DT': 1.0},
    'CONJ': {'PRO': 0.4, 'V': 0.3, 'N': 0.3}}
pword = {'PRO': {'I': 0.25, 'THAT': 0.25, 'THIS': 0.25, 'THEY': 0.25},
    'V': {'HOPE': 0.1, 'WORKS': 0.1, 'FISH': 0.1, 'CAN': 0.2, 'EATS': 0.1, 'SHOOTS': 0.1, 'LEAVES': 0.1, 'FLIES': 0.1, 'LEAVES': 0.1},
    'N': {'PANDA': 0.1, 'FISH': 0.2, 'CAN': 0.3, 'SHOOTS': 0.1, 'LEAVES': 0.1, 'TIME': 0.1, 'ARROW': 0.1},
    'CONJ': {'THAT': 0.5, 'AND': 0.5},
    'AUX': {'CAN': 1.0},
    'DT': {'AN': 0.35, 'A': 0.35, 'THIS': 0.3},
    'PREP': {'LIKE': 1.0},
    '</s>': {'</s>': 1.0}}

sentence = None
best = defaultdict(lambda : defaultdict(float))
best[0]["<s>"] = 1
back = defaultdict(dict)

def backtrack(i, tag):
    if i == 0:
        return []
    return backtrack(i-1, back[i][tag]) + [(sentence[i], tag)]

try:
    sentence = sys.argv[1].upper()
    sentence = ['<s>'] + sentence.split() + ['</s>']
    for i, word in enumerate(sentence[1:], 1):
        for tag in tags[word]:
            for prev in best[i-1]:
                if tag in ptag[prev]:
                    score = best[i-1][prev] * ptag[prev][tag] * pword[tag][word]
                    if score > best[i][tag]:
                        best[i][tag] = score
                        back[i][tag] = prev

    print(str(backtrack(len(sentence)-1, "</s>")[:-1]) + " # " + str(best[len(sentence)-1]['</s>']))
except:
	sys.stderr.write("usage: tagging.py <sentence>\n")
	sys.exit(1)
