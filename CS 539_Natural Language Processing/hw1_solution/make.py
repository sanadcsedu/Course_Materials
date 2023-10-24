#!/usr/bin/python
import sys

vocab_dict = {}

print "1"

for line in sys.stdin:
    prefix = ""
    chars = line.strip().split()
    idx_dict = vocab_dict
    for char in chars:
        if char not in idx_dict:
            if len(prefix) == 0:
                print "(0 ({} {}))".format(char, char)
            else:
                print "({} ({} {}))".format(prefix, prefix + char, char)
            idx_dict[char] = {}
        idx_dict = idx_dict[char]
        prefix = prefix + char
    print "({} (1 *e*))".format(prefix)
print "(1 (0 _))"
