from __future__ import print_function
import sys
import pdb
from collections import defaultdict
import math
import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class EM:
    def __init__(self):
        self.word_alignments = defaultdict(list)
        self.fractional_count = defaultdict(list)
        self.p_ep_jp = defaultdict(lambda: defaultdict(float))
        self.corpus = 1
        self.all_eprons = []

    def gen_alignment(self, i, j, jphon, align, epron):
        if i == 0 and j == len(jphon):
            temp = align.copy()
            self.word_alignments[epron].append(temp)
            return

        Ji = ""
        for l in range(j, min(len(jphon), j + 3)):
            if len(Ji) > 0:
                Ji += " " + str(jphon[l])
            else:
                Ji += str(jphon[l])
            align.append(Ji)
            self.gen_alignment(i - 1, l + 1, jphon, align, epron)
            align.pop()

    def initialize(self):
        total = 0
        for epron in self.word_alignments:
            _len = len(self.word_alignments[epron])
            self.fractional_count[epron] = [None] * _len
            total += _len

        cnt = 0

        for epron in self.word_alignments:
            _len = len(self.word_alignments[epron])
            cnt += _len

        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] = 1 / cnt

    def assign_alignment(self, epron, jpron):
        ephon = epron.split(' ')
        jphon = jpron.split(' ')
        align = []
        self.gen_alignment(len(ephon), 0, jphon, align, epron)

    def resetP(self):
        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] = 0

    def print_alignments(self, iter):
        nonzero = 0
        eprint("iteration {} ----- corpus prob= {}".format(iter, self.corpus))
        self.corpus = 0
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

    def get_epron_jpron_probs(self):
        for e in sorted(self.p_ep_jp):
            for w, value in sorted(self.p_ep_jp[e].items(), reverse=True, key=lambda kv: kv[1]):
                if value > 0.01:
                    print("{} : {} # {}".format(e, w, value))

    def m_step(self):
        #resets P(epron | jpron), which is the model probability
        self.resetP()

        # M-step: count-and-divide based on the collected fractional counts from all pairs to get new prob
        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] += self.fractional_count[epron][i]

            #removes alignments which are less than 0.01
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    if self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] < 0.01:
                        self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] = 0

    def isvalid(self, jpron, candidate):
        jphon = jpron.split(' ')
        jcand = []
        for i in range(len(candidate)):
            jl = candidate[i].split(' ')
            for j in jl:
                jcand.append(j)
        if jphon == jcand:
            return True
        return False

    def iterations(self, iter):
        sz = len(self.all_eprons)
        for i in range(sz):
            epron = self.all_eprons[i]

            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])

            for i in range(_len):
                self.fractional_count[epron][i] = 1
                #if self.isvalid(jpron, self.word_alignments[epron][i]):
                for j in range(len((self.word_alignments[epron][i]))):
                    self.fractional_count[epron][i] *= self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]]

            _sum = sum(self.fractional_count[epron])

            self.corpus += math.log(_sum)

            for i in range(len(self.word_alignments[epron])):
                self.fractional_count[epron][i] /= _sum

        self.m_step()
        self.print_alignments(iter)

    def run(self, iter):

        lines = sys.stdin.readlines()
        for i in range(0, len(lines), 3):
            epron = lines[i].rstrip()
            jpron = lines[i + 1].rstrip()
            skip = lines[i+2].rstrip()
            self.all_eprons.append(epron)
            self.assign_alignment(epron, jpron)

        self.initialize()

        for i in range(iter):
            self.iterations(i)

        self.get_epron_jpron_probs()


if __name__ == "__main__":
    # try:
    #     em_iterations = int(sys.argv[1])
    # except:
    #     sys.stderr.write("usage: em.py <# of Iterations>\n")
    #     sys.exit(1)

    start = time.clock()
    em = EM()
    # em.run(em_iterations)
    em.run(15)
    print("Time: {}".format(time.clock() - start))
