import sys
import pdb
from itertools import permutations
#This one uses Less than one smooting
class make_trigramAP:
    def __init__(self, lidstone):
        self.dict_freq = {}
        l1 = ["<s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"]
        l2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "</s>"]

        for i in range(len(l1)):

            if l1[i] not in self.dict_freq:
                self.dict_freq[l1[i]] = {}

            for j in range(len(l2)):

                if l2[j] not in self.dict_freq[l1[i]]:
                    self.dict_freq[l1[i]][l2[j]] = lidstone

    def update_freq(self, cur, new_state):

        self.dict_freq[cur][new_state] += 1

    def gen_wfsa(self):
        for A in sorted(self.dict_freq.keys()):
            sum = 0
            for B in sorted(self.dict_freq[A].keys()):
                sum += self.dict_freq[A][B]

            for B in sorted(self.dict_freq[A].keys()):
                print("({} ({}  {} {}))".format(A, B, B, self.dict_freq[A][B] / sum))


if __name__ == '__main__':

    f = open("train.txt", "r")
    mk_tri = make_trigramAP(0.5)

    print("</s>")
    print("(0 (<s> <s>))")

    for line in f:
        _str = str(line)
        prev = "<s>"
        for i in range(len(_str)):

            if _str[i].isalpha():
                mk_tri.update_freq(prev, _str[i])
                prev = _str[i]

            elif _str[i] == " ":
                mk_tri.update_freq(prev, "_")
                prev = "_"

        mk_tri.update_freq(prev, "</s>")

    mk_tri.gen_wfsa()
