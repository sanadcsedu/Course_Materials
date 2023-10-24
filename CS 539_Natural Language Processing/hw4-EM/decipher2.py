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

            example_split = ['<s>'] + [e for e in example] + ['</s>']
            n = len(example_split)

            forward = defaultdict(lambda: defaultdict(float))
            backward = defaultdict(lambda: defaultdict(float))

            # Forwards
            forward[0]['<s>'] = 1
            for i in range(1, n-1):
                emission = example_split[i]
                for e_last in forward[i-1]:
                    for e_next in CHARACTERS:
                        score = forward[i-1][e_last] * self.language_model[e_last][e_next] * self.emission_model[e_next][emission]
                        
                        forward[i][e_next] += score

            # Calculate p(x)
            for e_last in forward[n-2]:
                stop = '</s>'
                try:
                    score = forward[n-2][e_last] * self.language_model[e_last][stop]
                except:
                    pdb.set_trace()
                forward[n-1][stop] += score

            #print('back')
            # Backward
            backward[n-1]['</s>'] = 1
            for i in range(n-2, 0, -1):
                emission = example_split[i+1]
                for e_next in backward[i+1]:
                    for e_last in CHARACTERS:
                        if e_next == '</s>':
                            score = backward[i+1][e_next] * self.language_model[e_last][e_next]
                        else:
                            score = backward[i+1][e_next] * self.language_model[e_last][e_next] * self.emission_model[e_next][emission]

                        backward[i][e_last] += score
            
            # Calculate p(x)
            emission = example_split[1]
            for e_next in backward[1]:
                start = '<s>'
                try:
                    score = backward[1][e_next] * self.language_model[start][e_next] * self.emission_model[e_next][emission]
                except:
                    pdb.set_trace()
                backward[0][start] += score

            
            for t in range(1, n-1):
                for j in CHARACTERS:
                    try:
                        beta[j][example[t-1]] += (forward[t][j] * backward[t][j]) / forward[n-1]['</s>']
                    except:
                        print()

        for j in CHARACTERS:
            denom = sum(beta[j].values())
            for o in CHARACTERS:
                beta[j][o] /= denom
        
        self.emission_model = beta
        
        # print emissions
        # for c in self.emission_model:
        #     print(f'{c}:')
        #     for w, value in sorted(self.emission_model[c].items(), reverse=True, key=lambda kv: kv[1]):
        #         if (self.emission_model[c][w] > 0.0001):
        #             eprint("    {:^4s} {}: {}".format("", w, self.emission_model[c][w]))
        #
        
        #M step is inside print alignments
        #self.print_emissions(iter)

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
            print("Cipher : {}".format(example))
            print("Decoded: ", end="")
            for char in example:
                _max = -1
                argmax = ""
                for c in CHARACTERS:
                    if _max < self.emission_model[c][char]:
                        _max = self.emission_model[c][char]
                        argmax = c
                if argmax == '_':
                    argmax = ' '
                print(argmax, end="")
            print()

        #pdb.set_trace()

        #self.get_epron_jpron_probs()


if __name__ == "__main__":
    #try:
    #    em_iterations = int(sys.argv[1])
    #except:
    #    sys.stderr.write("usage: em.py <# of Iterations>\n")
    #    sys.exit(1)
    #start = time.clock()
    em = EM('bigram')
    # em.run(em_iterations)
    em.run(100)
    #print("Time: {}".format(time.clock() - start))
