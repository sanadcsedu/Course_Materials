import sys
import re
from collections import defaultdict

class viterbi_align:
    def __init__(self, epron_jpron_prob):
        self.word_alignments = defaultdict(list)
        self.p_ep_jp = defaultdict(lambda: defaultdict(float))
        self.fractional_count = defaultdict(list)
        self.errorcnt = 0
        f = open(epron_jpron_prob, "r")
        for line in f:
            text = str(line)
            results = re.split(' |:|#|\n', text)
            results = list(filter(None, results))
            temp = ""
            for i in range(1, len(results) - 1):
                if len(temp) > 0:
                    temp += " " + results[i]
                else:
                    temp += results[i]
            self.p_ep_jp[results[0]][temp] = float(results[len(results) - 1])

        f.close()

    def gen_alignment(self, i, j, jphon2, align, epron2):
        if i == 0 and j == len(jphon2):
            temp = align.copy()
            # if epron2 == "EH R":
            #     print(temp)
            self.word_alignments[epron2].append(temp)
            return

        Ji = ""
        for l in range(j, min(len(jphon2), j + 3)):
            if len(Ji) > 0:
                Ji += " " + str(jphon2[l])
            else:
                Ji += str(jphon2[l])
            # Ji += str(jphon2[l])
            align.append(Ji)
            self.gen_alignment(i - 1, l + 1, jphon2, align, epron2)
            align.pop()

    def assign_alignment(self, epron1, jpron1):
        ephon1 = epron1.split(' ')
        jphon1 = jpron1.split(' ')
        align = []
        # print(ephon)
        # print(jphon)
        self.gen_alignment(len(ephon1), 0, jphon1, align, epron1)

    def isvalid(self, jpron, candidate):
        jphon = jpron.split(' ')
        jcand = []
        for i in range(len(candidate)):
            jl = candidate[i].split(' ')
            for j in jl:
                # print(candidate[i][j])
                jcand.append(j)
        # print(jphon)
        # print(jcand)
        if jphon == jcand:
            return True
        return False

    def generate(self, epron, jpron):

        ephon = epron.split(' ')
        _len = len(self.word_alignments[epron])

        for i in range(_len):
            self.fractional_count[epron][i] = 1
            if self.isvalid(jpron, self.word_alignments[epron][i]):
                for j in range(len((self.word_alignments[epron][i]))):
                    # print("# {} {} {}".format(ephon[j], self.word_alignments[epron][i][j], self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]]))
                    if self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]] < 0.01:
                        self.fractional_count[epron][i] *= 1e-5
                    else:
                        self.fractional_count[epron][i] *= self.p_ep_jp[ephon[j]][self.word_alignments[epron][i][j]]
                    # print("Invalid")
            else:
                self.fractional_count[epron][i] = 0

        _max = -1
        argmax = -1
        for i in range(len(self.word_alignments[epron])):
            if self.fractional_count[epron][i] > _max:
                argmax = i
                _max = self.fractional_count[epron][i]

        # print(self.word_alignments[epron])
        # print(self.fractional_count[epron])
        # print(argmax)

        result = []
        ass = 1
        for i in range(len(self.word_alignments[epron][argmax])):
            zi = self.word_alignments[epron][argmax][i].split(' ')
            for j in range(len(zi)):
                result.append(ass)
            ass += 1

        print(epron)
        print(jpron)
        for i in range(len(result)):
            if i == 0:
                print(result[i], end="")
            else:
                print(" {}".format(result[i]), end="")
        print()


    def run(self):

        # self.assign_alignment("EH R", "E E R U")
        # self.fractional_count["EH R"] = [None] * len(self.word_alignments["EH R"])
        # self.generate("EH R", "E E R U")
        lines = sys.stdin.readlines()
        epron = []
        jpron = []
        _len = 0
        for i in range(0, len(lines), 3):
            ep = lines[i].rstrip()
            jp = lines[i + 1].rstrip()
            skip = lines[i + 2].rstrip()
            epron.append(ep)
            jpron.append(jp)
            _len += 1
            self.assign_alignment(ep, jp)

        # for i in range(_len):
        #     print(epron[i])
        #     print(jpron[i])
        #     print(self.word_alignments[epron[i]])

        total = 0
        for words in self.word_alignments:
            sz = len(self.word_alignments[words])
            self.fractional_count[words] = [None] * sz
            total += sz

        for i in range(_len):
            # if epron[i] == "EH R":
            #     print(epron[i])
            #     print(jpron[i])
            #     print(self.word_alignments[epron[i]])
            #     self.generate(epron[i], jpron[i])
            self.generate(epron[i], jpron[i])


if __name__ == "__main__":
    try:
        epron_jpron_prob = sys.argv[1]
    except:
        sys.stderr.write("usage: em.py <EM_Iterations(int)>\n")
        sys.exit(1)

    viterbi = viterbi_align(epron_jpron_prob)
    viterbi.run()
