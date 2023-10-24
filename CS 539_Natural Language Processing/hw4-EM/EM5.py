from __future__ import print_function
import sys
import pdb
from collections import defaultdict
import math
import time
import pickle
import operator

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

CHARACTERS = ['_','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

class EM:
    def __init__(self, language_model):
        self.language_model = load_obj(language_model)
        self.emission_model = defaultdict(lambda: defaultdict(float))
        for x in CHARACTERS:
            for y in CHARACTERS:
                self.emission_model[x][y] = 1/27
        self.all_examples = list()
        self.gamma = defaultdict(lambda: defaultdict(float))

    def print_alignments(self, iter):
    
        for e in sorted(self.p_ep_jp):
            eprint("{}|->".format(e), end="")
            total = 0
            for w in sorted(self.p_ep_jp[e]):
                total += self.p_ep_jp[e][w]
            for w, value in sorted(self.p_ep_jp[e].items(), reverse=True, key=lambda kv: kv[1]):
                self.p_ep_jp[e][w] = value / total
                if self.p_ep_jp[e][w] > 0.01:
                    eprint("{:^4s} {}: {:.2f}".format("", w, self.p_ep_jp[e][w]), end=" ")
                    nonzero += 1
            eprint()
        eprint("nonzeros = {}".format(nonzero))

    def iterations(self, iter):

        beta = defaultdict(lambda: defaultdict(float))
        for example in self.all_examples:

            n = len(example)

            forward = defaultdict(lambda: defaultdict(float))
            backward = defaultdict(lambda: defaultdict(float))

            # Forwards
            forward[0]['<s>'] = 1
            for i, emission in enumerate(example):
                for e_last in forward[i]:
                    for e_next in CHARACTERS:
                    print(f'  j = {j}')
                        try:
                            #print(f'    forward[{i}][{e_last}] * self.language_model[{e_last}][{e_next}] * self.emission_model[{e_next}][{emission}]')
                            score = forward[i][e_last] * self.language_model[e_last][e_next] * self.emission_model[e_next][emission]
                        except:
                            pdb.set_trace()

                        forward[i+1][e_next] += score

            # Calculate p(x)
            for e_last in forward[n]:
                stop = '</s>'
                try:
                    score = forward[n][e_last] * self.language_model[e_last][stop]
                except:
                    pdb.set_trace()
                forward[n+1][stop] += score

            # Backward
            backward[n+1]['</s>'] = 1
            for i, emission in reversed(list(enumerate(example))):
                for e_next in backward[i+2]:
                    for e_last in CHARACTERS:
                    # print(f'  j = {j}')
                        try:
                            score = backward[i+2][e_next] * self.language_model[e_last][e_next] *self.emission_model[e_last][emission]
                        except:
                            pdb.set_trace()

                        backward[i+1][e_last] += score
            
            # Calculate p(x)
            for e_last in backward[1]:
                start = '<s>'
                try:
                    score = backward[1][e_last] * self.language_model[start][e_last]
                except:
                    pdb.set_trace()
                backward[0][start] += score
            
            for t in range(1, n+1):
                for j in CHARACTERS:
                    try:
                        beta[j][example[t-1]] += (forward[t][j] * backward[t][j])/forward[n+1][stop]
                    except:
                        pdb.set_trace()
                        print()

        for j in CHARACTERS:
            denom = sum(beta[j].values())
            for o in CHARACTERS:
                beta[j][o] /= denom
        
        self.emission_model = beta
        
        #pdb.set_trace()
        
        #M step is inside print alignments
        #self.print_alignments(iter)

    def run(self, iter):

        #lines = sys.stdin.readlines()
        lines = ['gjkgcbkycnjpkovryrbgkcrvsczbkyhkahqvykgjvjkcpekrbkjxdjayrpmkyhkmhkyhkyvrcukrpkbjfjvcukihpygb',
            'oqykcykujcbykhpjkejihavcyrakvjmrijkjxrbybkrpkjfjvzkiclhvkarfrurtcyrhpkhvkaquyqvjkrpkygjkshvue',
            'auqobkrpkvjajpykzjcvbkgcfjkwhqpekohvcbkbhkerwwraquykyhkpjmhyrcyjksrygkygcykicpzkbgqpkgrbkaurjpyb',
            'gjkbcrekygjkoqvjcqkiqbykgcfjkygjkerbdqyjkvjbhufjekozkwjovqcvzkrpkhvejvkyhkdjvwhvikygjkajpbqb',
            'oqykygjkdqouraryzkbqvvhqperpmkygjkejocaujkajvycrpuzkbjvfjekyhkwhaqbkckbdhyurmgykhpkyghbjkjwwhvyb']

        for i in range(0, len(lines)):
            example = lines[i].rstrip()
            self.all_examples.append(example)

        for i in range(iter):
            self.iterations(i)

        for example in self.all_examples:
            print(f'{example}:')
            for char in example:
                print(max(self.emission_model[char].items(), key=operator.itemgetter(1))[0], end="")
            print()

        #pdb.set_trace()

        self.get_epron_jpron_probs()


if __name__ == "__main__":
    #try:
    #    em_iterations = int(sys.argv[1])
    #except:
    #    sys.stderr.write("usage: em.py <# of Iterations>\n")
    #    sys.exit(1)
    #start = time.clock()
    em = EM('bigram')
    # em.run(em_iterations)
    em.run(25)
    #print("Time: {}".format(time.clock() - start))
