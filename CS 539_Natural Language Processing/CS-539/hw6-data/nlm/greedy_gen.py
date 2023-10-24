#import torch
import nlm
nlm.NLM.load('large')
h = nlm.NLM()
for i in range(100):
    c, p = max(h.next_prob().items(), key=lambda x: x[1])
    print(c, "%.2f <- p(%s | ... %s)" % (p, c, " ".join(map(str, h.history[-4:]))))
    h += c

cat test.txt | sed -e 's/ /_/g;s/\(.\)/\1 /g' | awk '{printf("<s> %s </s>\n", $0)}' | carmel -sribI
