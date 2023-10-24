import sys
f = open("eword-epron.data", "r")
dict_freq = {}
trans ={}
#wordlist = [line.split()[0] for line in f]
wordlist = ["HELLO", "HELL", "HELMET", "WORLD"]
print("1")

for i in range(len(wordlist)):
    str = wordlist[i]
    prev = "0"
    for j in range(len(str)):
        if prev not in dict_freq:
            dict_freq[prev] = {}
            trans[prev] = {}
        if prev == "0":
            new_state = str[j]
        else:
            new_state = prev + str[j]

        if new_state not in dict_freq[prev]:
            dict_freq[prev][new_state] = 1
            trans[prev][new_state] = str[j]
        else:
            dict_freq[prev][new_state] += 1

        prev = new_state

    if prev not in dict_freq:
        dict_freq[prev] = {}
        trans[prev] = {}
    dict_freq[prev]["1"] = 1
    trans[prev]["1"] = "1"

for A in sorted(trans.keys()):
    sum = 0
    for B in sorted(trans[A].keys()):
        sum += dict_freq[A][B]
    for B in sorted(trans[A].keys()):
        if trans[A][B] == "1":
            print("({} ({} *e* {} {}))".format(A, B, A, dict_freq[A][B]/sum))
        else:
            print("({} ({} {} *e* {}))".format(A, B, trans[A][B], dict_freq[A][B]/sum))