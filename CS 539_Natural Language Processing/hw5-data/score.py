import sys

score = []
for line in sys.stdin:
    _, s, _ = line.split()
    score.append(float(s.strip()))

p = score[2] / score[0]
r = score[2] / score[1]
f1 = 2 * p * r / (p + r)
print("Precision: {}".format(p))
print("Recall: {}".format(r))
print("Precision: {}".format(f1))