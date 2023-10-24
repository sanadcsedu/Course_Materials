import sys
dict = {}
print("1")
for line in sys.stdin:
    temp = ""
    strr = str(line)
    prev = "0"
    for i in range(len(strr)):
        if strr[i].isalpha():
            temp += strr[i]
            if dict.get(temp, -1) == -1:
                dict.update({temp: 1})
                print("(" + prev + " (" + temp + " " + strr[i] + "))")
            prev = temp
    print("(" + prev + " (1 " + "*e*))")
print("(1 (0 _))")