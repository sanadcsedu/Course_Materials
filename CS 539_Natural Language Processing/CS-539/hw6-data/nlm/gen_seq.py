#import torch
import nlm
import sys
import math
nlm.NLM.load('base')
import random

for _ in range(10):
    h = nlm.NLM()
    chars = list(h.next_prob().keys())
    c = "<s>"
    t = 0.5
    choice = "<s>"
    while c != "</s>":
        probs = h.next_prob()
        # s = sum(p ** (1/t) for p in probs.values())
        probs = {c: p ** (1 / t) for (c, p) in probs.items()}
        print(choice, end = ' ')
        [choice] = random.choices(chars, [probs[c] for c in chars])
        h += choice
        if choice == "</s>":
            h = nlm.NLM()
            break
    print(choice)
    print()
