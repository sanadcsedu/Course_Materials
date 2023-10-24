#import torch
import nlm
import sys
import math
nlm.NLM.load('large')
h = nlm.NLM() #initialize a state (and observing <s>)
# p = 1
beam_len = 10

# lines = sys.stdin.readlines()

beam = []
# lines = "therestcanbeatotalmessandyoucanstillreaditwithoutaproblem"
lines = "thisisbecausethehumanminddoesnotreadeveryletterbyitselfbutthewordasawhole"
p = 1

beam.append((1,h))
for c in lines:
    temp = []
    for (p, h) in beam:
        prob1 = p * h.next_prob("_")
        h1 = h + "_" + c
        temp.append((prob1, h1))

        prob2 = p * h.next_prob(c)
        h2 = h + c
        temp.append((prob2, h2))

    beam = sorted(temp, reverse = True)[:beam_len]

p, h = beam[0]
print(("".join(h.history)).replace("_", " ").replace("<s>", ""))