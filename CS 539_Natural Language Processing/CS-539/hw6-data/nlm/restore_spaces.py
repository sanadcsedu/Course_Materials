#import torch
import nlm
import sys
import math
nlm.NLM.load('large')

beam_len = 20

sys.stdout = open("spaces_restored_large.txt", 'w')

with open("test.txt.nospaces") as fp:
    for lines in fp:
        lines = lines.strip()
        h = nlm.NLM()
        beam = [(0, h)]
        for c in lines:
            temp = []
            for (p, h) in beam:
                prob1 = p + math.log(h.next_prob("_"))
                h1 = h + "_"
                prob1 += math.log(h1.next_prob(c))
                h1 = h1 + c
                temp.append((prob1, h1))

                prob2 = p + math.log(h.next_prob(c))
                h2 = h + c
                temp.append((prob2, h2))

            beam = sorted(temp, reverse=True)[:beam_len]

        p, h = beam[0]
        print(("".join(h.history)).replace("_", " ").replace("<s>", ""))
