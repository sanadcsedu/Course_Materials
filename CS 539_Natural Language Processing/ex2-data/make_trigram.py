import sys
import pdb
class make_trigram:
    def __init__(self):
        self.dict = {}
        self.dict_freq = {}

    def update_freq(self, cur, new_state, trans):
        if cur not in self.dict:
            self.dict[cur] = {}
            self.dict_freq[cur] = {}

            if new_state not in self.dict[cur]:
                self.dict[cur][new_state] = {}
                self.dict_freq[cur][new_state] = {}

                if trans not in self.dict[cur][new_state]:
                    self.dict[cur][new_state][trans] = trans
                    self.dict_freq[cur][new_state][trans] = 1
                else:
                    self.dict_freq[cur][new_state][trans] += 1

            else:
                if trans not in self.dict[cur][new_state]:
                    self.dict[cur][new_state][trans] = trans
                    self.dict_freq[cur][new_state][trans] = 1
                else:
                    self.dict_freq[cur][new_state][trans] += 1

        else:
            if new_state not in self.dict[cur]:
                self.dict[cur][new_state] = {}
                self.dict_freq[cur][new_state] = {}

                if trans not in self.dict[cur][new_state]:
                    self.dict[cur][new_state][trans] = trans
                    self.dict_freq[cur][new_state][trans] = 1
                else:
                    self.dict_freq[cur][new_state][trans] += 1

            else:
                if trans not in self.dict[cur][new_state]:
                    self.dict[cur][new_state][trans] = trans
                    self.dict_freq[cur][new_state][trans] = 1
                else:
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
    mk_tri = make_trigram()

    print("</s>")
    print("(0 (<s> <s>))")

    for line in f:
        _str = str(line)
        prev1 = "<s>"
        past = "<s>"
        for i in range(len(_str)):

            if _str[i].isalpha():
                new_state = prev1 + _str[i]
                mk_tri.update_freq(past, new_state, _str[i])
                past = new_state
                prev1 = _str[i]

            elif _str[i] == " ":
                new_state = prev1 + "_"
                mk_tri.update_freq(past, new_state, "_")
                past = new_state
                prev1 = "_"

        mk_tri.update_freq(past, "</s>", "</s>")

    mk_tri.gen_wfsa()
