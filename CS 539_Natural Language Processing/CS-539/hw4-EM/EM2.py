from __future__ import print_function
import sys
import pdb
from collections import defaultdict
import math


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class EM:
    def __init__(self):
        self.word_alignments = defaultdict(list)
        self.fractional_count = defaultdict(list)
        self.x_pair = defaultdict(str)
        self.p_ep_jp = defaultdict(lambda: defaultdict(float))
        self.corpus = 1
        self.all_epron = []
        self.all_jpron = []

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

    def init(self):
        total = 0
        for epron in self.word_alignments:
            _len = len(self.word_alignments[epron])
            self.fractional_count[epron] = [0] * _len
            total += _len

        cnt = 0

        for epron in self.word_alignments:
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    cnt = cnt + 1
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
        self.x_pair[epron] = jpron
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
        # resets P(epron | jpron), which is the model probability
        self.resetP()

        # M-step: count-and-divide based on the collected fractional counts from all pairs to get new prob
        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] += self.fractional_count[epron][i]

            # removes alignments which are less than 0.01
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    if self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] < 0.01:
                        self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] = 0

    def iterations(self, iter):

        for epron in self.all_epron:

            forward = defaultdict(lambda: defaultdict(float))
            backward = defaultdict(lambda: defaultdict(float))

            eprons = epron.split(' ')
            jprons = self.x_pair[epron].split(' ')

            n, m = len(eprons), len(jprons)

            # Forwards
            # forward[0][0] = 1
            for i in range(0, n+1):
                forward[0][i] = 1

            for i in range(0, n):
                epho = eprons[i]
                # print(f'i = {i}, epho = {epho}')
                for j in forward[i]:
                    # print(f'  j = {j}')
                    for k in range(1, min(m - j, 3) + 1):
                        jseg = ' '.join(jprons[j:j + k])
                        score = forward[i][j] * self.p_ep_jp[epho][jseg]
                        forward[i + 1][j + k] += score

            # totalprob *= forward[n][m]

            # Backwards
            for i in range(0, n + 1):
                backward[i][m] = 1
            for i in range(n, 0, -1):
                epho = eprons[i - 1]
                # print(f'i = {i}, epho = {epho}')
                for j in backward[i]:
                    # print(f'  j = {j}')
                    for k in range(1, min(j, 3) + 1):
                        jseg = ' '.join(jprons[j - k:j])
                        score = backward[i][j] * self.p_ep_jp[epho][jseg]
                        backward[i - 1][j - k] += score

            # beta *= backward[0][0]

            # regenerate fractional counts
            for z, alignment_z in enumerate(self.word_alignments[epron]):
                last_j = 0
                # print(f'alignment = {alignment_z}:')
                for u in range(len(eprons)):
                    epho = eprons[u]
                    jseg = alignment_z[u]
                    next_j = last_j + len(jseg) - (len(jseg) // 2)
                    self.fractional_count[epron][z] += (forward[u][last_j] * backward[u + 1][next_j] *
                                                        self.p_ep_jp[epho][jseg]) / forward[n][m]
                    last_j = next_j

            _sum = sum(self.fractional_count[epron])

            self.corpus += math.log(_sum)

            for i in range(len(self.word_alignments[epron])):
                self.fractional_count[epron][i] /= _sum
            # print(self.fractional_count[epron])

        self.m_step()
        self.print_alignments(iter)

    def run(self, iter):

        lines = sys.stdin.readlines()
        for i in range(0, len(lines), 3):
            epron = lines[i].rstrip()
            jpron = lines[i + 1].rstrip()
            skip = lines[i + 2].rstrip()
            if epron == 'PAUSE':
                continue
            self.all_epron.append(epron)
            self.all_jpron.append(jpron)
            self.assign_alignment(epron, jpron)
        self.init()
        # print(cnt)
        for i in range(iter):
            self.iterations(i)

        self.get_epron_jpron_probs()


if __name__ == "__main__":
    # try:
    #     em_iterations = int(sys.argv[1])
    # except:
    #     sys.stderr.write("usage: em.py <# of Iterations>\n")
    #     sys.exit(1)

    em = EM()
    # em.run(em_iterations)
    em.run(5)
