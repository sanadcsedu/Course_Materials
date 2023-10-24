import sys
f = open("train.txt", "r")
dict_count = {}
#for line in sys.stdin:
nm = 0
cnt = 0
for line in f:
    strr = str(line)
    cnt += 1
    for i in range(len(strr)):
        nm += 1
        if strr[i].isalpha():
            if dict_count.get(strr[i], -1) == -1: #character exists in the dictionary
                dict_count.update({strr[i]: 1})
            else:
                dict_count[strr[i]] += 1

        if strr[i] == " ":
            if dict_count.get("_", -1) == -1:  # character exists in the dictionary
                dict_count.update({"_": 1})
            else:
                dict_count["_"] += 1

    dict_count.update({"</s>": cnt})

print("F")
print("(0 (1 <s>))")
# for idx, (letter, freq) in enumerate(sorted (dict_count.items())):
#     if letter == "</s>":
#         print("(1 (F  {} {}))".format(letter, freq / nm))
#     else:
#         print("(1 (1  {} {}))".format(letter, freq/nm))

for idx, (letter, freq) in enumerate(sorted (dict_count.items())):
    if letter == "</s>":
        print("(1 (F  {} {}))".format(letter, freq))
    else:
        print("(1 (1  {} {}))".format(letter, freq))

