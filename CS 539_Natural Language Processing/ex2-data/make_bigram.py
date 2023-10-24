import sys
import pdb
f = open("train.txt", "r")
#alphabet = "abcdefghijklmnopqrstuvwxyz_"

print("</s>")
print("(0 (<s> <s>))")
dict = {}
#for line in sys.stdin:
nm = 0
cnt = 0
for line in f:
    strr = str(line)
    cnt += 1
    prev = "<s>"
    for i in range (len(strr)):

        if strr[i].isalpha():

            if prev not in dict:
                dict[prev] = {}
                dict[prev][strr[i]] = 1
            else:
                if strr[i] not in dict[prev]:
                    dict[prev][strr[i]] = 1
                else:
                    dict[prev][strr[i]] += 1
            prev = strr[i]

        if strr[i] == " ":

            if prev not in dict:
                dict[prev] = {}
                dict[prev]["_"] = 1
            else:
                if "_" not in dict[prev]:
                    dict[prev]["_"] = 1
                else:
                    dict[prev]["_"] += 1
            prev = "_"

        #pdb.set_trace()

    if prev not in dict:
        dict[prev] = {}
        dict[prev]["</s>"] = 1
    else:
        if "</s>" not in dict[prev]:
            dict[prev]["</s>"] = 1
        else:
            dict[prev]["</s>"] += 1

#dict_count = sorted(dict_count.keys())
for A in sorted(dict.keys()):
    sum = 0
    for B in sorted(dict[A].keys()):
        sum += dict[A][B]
    for B in sorted(dict[A].keys()):
        print("({} ({} {} {} ))" .format(A, B, B, dict[A][B]/sum))
