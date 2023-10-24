import sys
from tree import Tree
from collections import defaultdict
from copy import deepcopy
from debinarize import deBinarize

def get_pcfg_dict():
    pcfg_dict = defaultdict(float)
    non_terminals = set()
    all_pairs = set()
    
    lines = open(sys.argv[1]).readlines()
    start_state = lines[0].strip()
    for line in lines[1:]:
        p1, p2_prob = line.split(" -> ")
        p2, prob = p2_prob.split(" # ")
        prob = float(prob)
        
        non_terminals.add(p1)
        all_pairs.add((p1,p2))
        
        pcfg_dict[(p1, p2)] = prob
#     print(all_pairs)
    return pcfg_dict, non_terminals, all_pairs, start_state

def cky(words):
    score = defaultdict(float)
    back = defaultdict()
    terminals = defaultdict()
    len_words = len(words)
    
    orgin_words = deepcopy(words)
#     print(len(sys.argv))
    
    if len(sys.argv) > 2:
        train_txt = [x.strip() for x in open(sys.argv[2]).readlines()]
        for i in range(len_words):
            if words[i] not in train_txt:
                words[i] = "<unk>"
#     print(orgin_words)
#     print(words)
    for i in range(len_words):
        for nt in non_terminals:
            word = words[i]
#             print(nt, word)
            if (nt, word) in all_pairs:
                score[(i, i + 1, nt)] = pcfg_dict[(nt, word)]
                terminals[(i, i + 1, nt)] = word
        for nt_1 in non_terminals:
            for nt_2 in non_terminals:
                if (nt_1, nt_2) in all_pairs:
                    prob = pcfg_dict[(nt_1, nt_2)] * score[(i, i + 1, nt_2)]
        
                    if prob > score[(i, i + 1, nt_1)]:
                        score[(i, i + 1, nt_1)] = prob
                        back[(i, i + 1, nt_1)] = (nt_2,)

    for i in range(1, len_words + 1):
        for j in range(len_words - i + 1):
            k = j + i
            for split in range(j + 1, k):
                for p1, p2 in all_pairs:
                    p2_split = p2.split()
                    if len(p2_split) > 1:
                        p2_1 = p2_split[0].strip()
                        p2_2 = p2_split[1].strip()
                        prob = score[(j, split, p2_1)] * score[(split, k, p2_2)] * pcfg_dict[(p1, p2)]

                        if prob > score[(j, k,  p1)]:
                            score[(j, k, p1)] = prob
                            back[(j, k, p1)] = (split, p2_1, p2_2)


            for nt_1 in non_terminals:
                for nt_2 in non_terminals:
                    if (nt_1, nt_2) in all_pairs:
                        prob = pcfg_dict[(nt_1, nt_2)] * score[(j, k, nt_2)]

                        if prob > score[(j, k, nt_1)]:
                            score[(j, k, nt_1)] = prob
                            back[(j, k, nt_1)] = (nt_2,)
                            
#     print(back)
    def backtrack(begin, end, label):
        
        if (begin, end, label) not in back:
            if (begin, end, label) in terminals:
                
                word = orgin_words[begin]
                t = Tree(label = label, subs = None, wrd = word, span = (begin, end))

            return t
        branches = back[(begin, end, label)]
        if len(branches) == 1:
            t1 = backtrack(begin, end, branches[0])
            
            t = Tree(label = label, subs = [t1], wrd = None, span = t1.span)
            return t

        elif len(branches) > 1:
            split, left, right = branches
            
            t1 = backtrack(begin, split, left)   
            t2 = backtrack(split, end, right)

            span_low = t1.span[0]
            span_high = t2.span[1]
            
            t = Tree(label = label, subs = [t1, t2], wrd = None, span = (span_low, span_high))
            return t
    
    if (0, len_words, start_state) not in back:
#         print(1)
        return "NONE" 
    
    return deBinarize(backtrack(0, len_words, start_state))
#     return backtrack(0, len_words, start_state)               
if __name__ == "__main__":
    pcfg_dict, non_terminals, all_pairs, start_state = get_pcfg_dict()
#     print pcfg_dict
#     print non_terms
#     print all_pairs
    for line in sys.stdin:
        tree = cky(line.strip().split())
        print tree