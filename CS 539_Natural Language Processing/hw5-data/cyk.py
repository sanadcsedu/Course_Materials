import sys
#from tree import Tree
import pdb, re
from collections import defaultdict


class PCFG:
    def __init__(self, pcfg_file):
    
        self.backtrack = None
    
        self.lexical_rules = defaultdict(lambda: defaultdict(float))
        self.unary_rules = defaultdict(lambda: defaultdict(float))
        self.binary_rules = defaultdict(lambda: defaultdict(float))
    
        with open(pcfg_file, 'r') as f:
            f.readline()
            for line in f:
                line = line.strip()
                g = re.match('(.*) -> (.*) # ([01]\.[0-9]*)', line)
                if re.match('[^A-Z]+', g.group(2)) or g.group(2) == 'I':
                    self.lexical_rules[g.group(1)][g.group(2)] = float(g.group(3))
                elif ' ' in g.group(2):
                    self.binary_rules[g.group(1)][g.group(2)] = float(g.group(3))
                else:
                    self.unary_rules[g.group(1)][g.group(2)] = float(g.group(3))

    def parse(self, sentence):
    
        #bestTree = None
        sentence = sentence.split(' ')
        n = len(sentence)
        score = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.backtrack = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Base case
        # Go over whole sequence
        #print('diff = {}'.format(1))
        for i, word in enumerate(sentence):
            for pos_tag in self.lexical_rules:
                if self.lexical_rules[pos_tag][word] > 0:
                    #print('\tscore[{}][{}][{}] = self.lexical_rules[{}][{}]'.format(pos_tag,i,i+1,pos_tag,word))
                    #print('\tscore[{}][{}][{}] = {}'.format(pos_tag,i,i+1,self.lexical_rules[pos_tag][word]))
                    score[pos_tag][i][i+1] = self.lexical_rules[pos_tag][word]
                    self.backtrack[pos_tag][i][i+1] = (word, -1)
                    
            newRuleAdded = True
            while newRuleAdded:
                for x in self.unary_rules:
                    for y in self.unary_rules[x]:
                        if self.unary_rules[x][y] > 0:
                            unaryScore = self.unary_rules[x][y] * score[y][i][i+1]
                            if unaryScore > score[x][i][i+1]:
                                #print('\tscore[{}][{}][{}] = self.unary_rules[{}][{}] * score[{}][{}][{}]'.format(x,i,i+1,x,y,y,i,i+1))
                                #print('\tscore[{}][{}][{}] = {} * {}'.format(x,i,i+1,self.unary_rules[x][y],score[y][i][i+1]))
                                score[x][i][i+1] = unaryScore
                                self.backtrack[x][i][i+1] = (y, 0)
                            else:
                                newRuleAdded = False

        for diff in range(2, n+1):
            #print('diff = {}'.format(diff))
            for i in range(n-diff+1):
                j = i + diff
                for x in self.binary_rules:
                    for yz in self.binary_rules[x]:
                        y, z = yz.split(' ')
                        for k in range(i+1,j):
                            if score[y][i][k] > 0 and score[z][k][j] > 0:
                                #print('\tscore[{}][{}][{}] = max(score[{}][{}][{}], p({}->{} {})*score[{}][{}][{}]*score[{}][{}][{}])'.format(x,i,j, x,i,j, x,y,z, y,i,k, z,k,j))
                                #print('\tscore[{}][{}][{}] = max({}, {} * {} * {})'.format(x,i,j,score[x][i][j], self.binary_rules[x][yz], score[y][i][k], score[z][k][j]))
                                splitScore = self.binary_rules[x][yz] * score[y][i][k] * score[z][k][j]
                                if splitScore > score[x][i][j]:
                                    score[x][i][j] = splitScore
                                    self.backtrack[x][i][j] = (yz, k)
                                    
                newRuleAdded = True
                while newRuleAdded:
                    for x in self.unary_rules:
                        for y in self.unary_rules[x]:
                            if self.unary_rules[x][y] > 0:
                                unaryScore = self.unary_rules[x][y] * score[y][i][j]
                                if unaryScore > score[x][i][j]:
                                    #print('\tscore[{}][{}][{}] = self.unary_rules[{}][{}] * score[{}][{}][{}]'.format(x,i,j,x,y,y,i,j))
                                    #print('\tscore[{}][{}][{}] = {} * {}'.format(x,i,j,self.unary_rules[x][y],score[y][i][j]))
                                    score[x][i][j] = unaryScore
                                    self.backtrack[x][i][j] = (y, 0)
                                else:
                                    newRuleAdded = False
        
        print('(TOP {})'.format(self.buildTree('S', 0, n)))
        
    def buildTree(self, a, i, j):
        rule = self.backtrack[a][i][j]
        try:
            if rule[1] == -1: # Terminal
                return '({} {})'.format(a, rule[0])
            elif rule[1] == 0: # Unary
                return '({} {})'.format(a, self.buildTree(rule[0],i,j))
            else: # Binary
                y,z = rule[0].split(' ')
                k = rule[1]
                return '({} {} {})'.format(a, self.buildTree(y,i,k), self.buildTree(z,k,j))
        except:
            pdb.set_trace()
            print()

def main():
    
    pcfg = PCFG('toy.pcfg.bin')

    for line in ['the boy saw a girl', 'I need to arrive early today']:
        pcfg.parse(line)

if __name__ == "__main__":
    main()