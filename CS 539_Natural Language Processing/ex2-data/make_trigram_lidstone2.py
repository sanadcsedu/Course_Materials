import sys
import pdb
from itertools import permutations
#This one uses Less than one smooting
class make_trigramAP:
    def __init__(self, lidstone):
        self.dict = {}
        self.dict_freq = {}
        l1 = ["<s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"]
        l2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"]
        l3 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "</s>"]

        for i in range(len(l2)):
            if "<s>" not in self.dict:
                self.dict["<s>"] = {}
                self.dict_freq["<s>"] = {}

            for j in range(len(l2)):
                new_state = "<s>" + l2[j]
                if new_state not in self.dict["<s>"]:
                    self.dict["<s>"][new_state] = {}
                    self.dict_freq["<s>"][new_state] = {}

                if l2[j] not in self.dict["<s>"][new_state]:
                    self.dict["<s>"][new_state][l2[j]] = l2[j]
                    self.dict_freq["<s>"][new_state][l2[j]] = lidstone

        for i in range(len(l1)):

            for j in range(len(l2)):

                curr_state = l1[i] + l2[j]
                if curr_state not in self.dict:
                    self.dict[curr_state] = {}
                    self.dict_freq[curr_state] = {}
                for k in range(len(l3)):

                    if l3[k] == "</s>":
                        if l3[k] not in self.dict[curr_state]:
                            self.dict[curr_state][l3[k]] = {}
                            self.dict_freq[curr_state][l3[k]] = {}
                            if l3[k] not in self.dict[curr_state][l3[k]]:
                                self.dict[curr_state][l3[k]][l3[k]] = l3[k]
                                self.dict_freq[curr_state][l3[k]][l3[k]] = lidstone
                        continue

                    new_state = l2[j] + l3[k]
                    if new_state not in self.dict[curr_state]:
                        self.dict[curr_state][new_state] = {}
                        self.dict_freq[curr_state][new_state] = {}

                    if l3[k] not in self.dict[curr_state][new_state]:
                        self.dict[curr_state][new_state][l3[k]] = l3[k]
                        self.dict_freq[curr_state][new_state][l3[k]] = lidstone

    def update_freq(self, cur, new_state, trans):

        self.dict_freq[cur][new_state][trans] += 1

    def gen_wfsa(self):
        for A in sorted(self.dict.keys()):
            sum = 0
            for B in sorted(self.dict[A].keys()):
                for C in sorted(self.dict[A][B].keys()):
                    sum += self.dict_freq[A][B][C]

            for B in sorted(self.dict[A].keys()):
                for C in sorted(self.dict[A][B].keys()):
                    print("({} ({}  {} {}))".format(A, B, C, self.dict_freq[A][B][C] / sum))


if __name__ == '__main__':

    f = open("train.txt", "r")
    lidstone = sys.argv[0]
    mk_tri = make_trigramAP(lidstone)
    print("Lidstone " + str(lidstone))

    print("</s>")
    print("(0 (<s> <s>))")

    for line in f:
        _str = str(line)
        prev = "<s>"
        for i in range(len(_str)):

            if _str[i].isalpha():
                mk_tri.update_freq(prev, _str[i], _str[i])
                prev = _str[i]

            elif _str[i] == " ":
                mk_tri.update_freq(prev, "_", "_")
                prev = "_"

        mk_tri.update_freq(prev, "</s>", "</s>")

    mk_tri.gen_wfsa()
