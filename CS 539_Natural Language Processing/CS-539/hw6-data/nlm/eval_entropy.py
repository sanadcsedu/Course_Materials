#import torch
import nlm
import sys
import math
nlm.NLM.load('large')
# h = nlm.NLM() #initialize a state (and observing <s>)
# p = 1
lines = sys.stdin.readlines()
entropy = 0
cnt = 0
for x in lines:

    line = [c for c in x.strip()] + ['</s>']
    h = nlm.NLM()
    p = 1
    # cnt = 0
    entropy_line = 1
    for c in line:
        if c == ' ':
            c = '_'
        entropy_line *= h.next_prob(c)  # include prob ( C | ...)
        h += c
        cnt += 1
    # print(p, entropy_line)
    entropy -= math.log(entropy_line,2)
# print(len(lines))
print(entropy/cnt)