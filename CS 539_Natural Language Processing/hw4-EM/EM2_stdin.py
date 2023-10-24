import sys
import pdb
from collections import defaultdict

class EM:
    def __init__(self):
        self.word_alignments = defaultdict(list)
        self.fractional_count = defaultdict(list)

    def gen_alignment(self, i, j, jphon, align, epron):
        if i == 0 and j == len(jphon):
            temp = align.copy()
            self.word_alignments[epron].append(temp)
            return

        Ji = ""
        for l in range(j, min(len(jphon), j+3)):
            Ji += str(jphon[l])
            align.append(Ji)
            self.gen_alignment(i-1, l+1, jphon, align, epron)
            align.pop()

    def assign_alignment(self, epron, jpron):
        ephon = epron.split(' ')
        jphon = jpron.split(' ')
        align = []
        self.gen_alignment(len(ephon), 0, jphon, align, epron)
        
        for epron in self.word_alignments:
            _len = len(self.word_alignments[epron])
            for alis in self.word_alignments[epron]:
                self.fractional_count[epron].append(1/_len)

    def iterations(self): #Calculates Fractional Count and P (Japanese | English)

        temp = defaultdict(lambda : defaultdict(float))

        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    temp[ephon[j]][self.word_alignments[epron][i][j]] += self.fractional_count[epron][i]

        # Removing alignments with less than 0.01 probability
        for epron in self.word_alignments:
            ephon = epron.split(' ')
            _len = len(self.word_alignments[epron])
            for i in range(_len):
                for j in range(len(self.word_alignments[epron][i])):
                    if temp[ephon[j]][self.word_alignments[epron][i][j]] < 0.01:
                        temp[ephon[j]][self.word_alignments[epron][i][j]] = 0


        nonzero = 0
        for e in sorted(temp):
            print("{}|->".format(e), end="    ")
            sum = 0
            for w in sorted(temp[e]):
                sum += temp[e][w]

            for w, value in sorted(temp[e].items(), reverse=True, key=lambda kv:kv[1]):
                if temp[e][w] > 0:
                    temp[e][w] = value / sum
                    print("# {}: {:.2f}".format(w, value), end=" ")
                    nonzero += 1
            print()
        print("nonzeros = {}".format(nonzero))


        #regenerate p(x, z)
        for epron in self.word_alignments:
            ephon = epron.split(' ')

            for i in range(len(self.word_alignments[epron])):
                self.fractional_count[epron][i] = 1
                for j in range(len((self.word_alignments[epron][i]))):
                    self.fractional_count[epron][i] *= temp[ephon[j]][self.word_alignments[epron][i][j]]

        #Normalize fractional Counts again
        for epron in self.word_alignments:
            sum = 0
            for i in range(len(self.word_alignments[epron])):
                sum += self.fractional_count[epron][i]

            for i in range(len(self.word_alignments[epron])):
                self.fractional_count[epron][i] /= sum

    def run(self, iter):

        lines = sys.stdin.readlines()
        for i in range(0, len(lines), 3):
            epron = lines[i].rstrip()
            jpron = lines[i+1].rstrip()
            if epron == "PAUSE":
                continue
            
            self.assign_alignment(epron, jpron)
        
        for i in range(iter):
            print("iteration {} ----- corpus prob= Not Implemented".format(i))
            self.iterations()
            print()


if __name__ == "__main__":
    try:
        em_iterations = int(sys.argv[1])
    except:
        sys.stderr.write("usage: em.py <EM_Iterations(int)>\n")
        sys.exit(1)

    em = EM()
    em.run(em_iterations)