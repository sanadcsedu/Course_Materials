import sys

class estimate:
    def __init__(self):
        self.dict_freq = {}

    def update(self, english_phoneme, japanese_phoneme, mapping):

        jedited = ["" for x in range(len(english_phoneme))]
        for i in range(len(mapping)):
            idx = int(mapping[i]) - 1
            if jedited[idx] == "":
                jedited[idx] = japanese_phoneme[i]
            else:
                jedited[idx] += " " + japanese_phoneme[i]

        for i in range(len(english_phoneme)):
            if english_phoneme[i] not in self.dict_freq:
                self.dict_freq[english_phoneme[i]] = {}

            if jedited[i] not in self.dict_freq[english_phoneme[i]]:
                self.dict_freq[english_phoneme[i]][jedited[i]] = 1
            else:
                self.dict_freq[english_phoneme[i]][jedited[i]] += 1

    def print_prob(self):

        for A in sorted(self.dict_freq.keys()):
            sum = 0
            for B in sorted(self.dict_freq[A].keys()):
                sum += self.dict_freq[A][B]

            for B in sorted(self.dict_freq[A].keys()):
                print("{} : {} # {}".format(A, B, self.dict_freq[A][B] / sum))

    def print_prob_WO(self, prob):

        for A in sorted(self.dict_freq.keys()):
            sum = 0
            for B in sorted(self.dict_freq[A].keys()):
                sum += self.dict_freq[A][B]

            theta = prob * sum
            sum = 0
            for B in sorted(self.dict_freq[A].keys()):
                if self.dict_freq[A][B] < theta:
                    self.dict_freq[A].pop(B, None)
                else:
                    sum += self.dict_freq[A][B]


            for B in sorted(self.dict_freq[A].keys()):
                print("{} : {} # {}".format(A, B, self.dict_freq[A][B] / sum))


if __name__ == "__main__":

    f = open("epron-jpron.data", "r")
    est = estimate()

    line1 = str(f.readline())
    while line1:
        line2 = str(f.readline())
        line3 = str(f.readline())

        english_phoneme = line1.strip().split(' ')
        japanese_phoneme = line2.strip().split(' ')
        mapping = line3.strip().split(' ')
        est.update(english_phoneme, japanese_phoneme, mapping)
        line1 = str(f.readline())

    #est.print_prob()
    est.print_prob_WO(0.01)

