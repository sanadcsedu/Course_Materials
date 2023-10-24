import sys
from tree import Tree
import pdb, re
from collections import defaultdict

count = defaultdict(lambda: defaultdict(float))

def countRules(tree):
    count[tree.label][' '.join([child.label for child in tree.subs])] += 1
    for child in tree.subs:
        if child.subs:
            countRules(child)
        else:
            count[child.label][child.word] += 1

def main():
    for line in sys.stdin:
        line = line.strip()
        countRules(Tree.parse(line))
            
    # Build grammar given counts
    binary_rule_count = 0
    unary_rule_count = 0
    lexical_rule_count = 0
    print 'TOP'
    for key in count:
        total = sum(count[key].values())
        for produces in count[key]:
            # Count new rule type
            if re.match('[^A-Z]+', produces):
                lexical_rule_count = lexical_rule_count + 1
            elif len(produces.split(' ')) == 2:
                binary_rule_count = binary_rule_count + 1
            elif len(produces.split(' ')) == 1:
                unary_rule_count = unary_rule_count + 1
        
            print '{} -> {} # {:.4f}'.format(key, produces, count[key][produces] / total)
            
    sys.stderr.write("Binary Rules: {}\n".format(binary_rule_count))
    sys.stderr.write("Unary Rules: {}\n".format(unary_rule_count))
    sys.stderr.write("Lexical Rules: {}\n".format(lexical_rule_count))

if __name__ == "__main__":
    main()