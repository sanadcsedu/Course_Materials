#import torch
import nlm
nlm.NLM.load('large')
h = nlm.NLM() #initialize a state (and observing <s>)
p = 1
for c in 't h e _ e n d _ '.split():
    print(p, h) # Cumulative probability and current state
    p *= h.next_prob(c) #include prob ( C | ...)
    h += c #Observe another character (Changing NLM state internally)
