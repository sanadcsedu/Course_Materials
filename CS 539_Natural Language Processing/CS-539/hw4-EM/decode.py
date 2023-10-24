import sys
from collections import defaultdict

START, END = ("<s>", "</s>")

def backtrack(back, i, e1, e):
    if i == 1:
        return [e]
    elif i <= 0:
        return []

    e2, j = back[i][e1][e]
    return backtrack(back, i-j, e2, e1) + [e]

p3epron = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
pjpron = defaultdict(lambda: defaultdict(float))
eprons = set([START, END])
with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2:
    for line in f1:
        l = line.split()
        p3epron[l[0]][l[1]][l[3]] = float(l[5])
    for line in f2:
        l = line.split(':')
        epron = l[0].strip()
        l = l[1].split('#')
        jprons = l[0].strip()
        pjpron[jprons][epron] = float(l[1].strip())
        eprons.add(epron)

for line in sys.stdin:
    best = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    back = defaultdict(lambda: defaultdict(dict))

    jprons_input = [START] + line.split() + [END]*2
    best[0][START][START] = 1.0
    pjpron[END][END] = 1.0
    for e in eprons:
        p3epron[e][END][END] = 1.0
        if e in p3epron[START][START] and e in pjpron[jprons_input[1]]:
            best[1][START][e] = p3epron[START][START][e] * pjpron[jprons_input[1]][e]
    for i in range(2, len(jprons_input)):
        for j in range(1, 4):
            jprons = ' '.join(jprons_input[i - j + 1:i + 1])
            for e in pjpron[jprons]:
                for e2 in best[i-j]:
                    for e1 in best[i-j][e2]:
                        if e in p3epron[e2][e1]:
                            score = best[i-j][e2][e1] * p3epron[e2][e1][e] * pjpron[jprons][e]
                            if score > best[i][e1][e]:
                                best[i][e1][e] = score
                                back[i][e1][e] = (e2, j)

    result = backtrack(back, len(jprons_input) - 1, END, END)[:-2]
    print ' '.join(result), '#', "%.6e" % best[len(jprons_input)-1][END][END]
