import sys
from tree import Tree

def binarize(tree):
    if tree.subs is None:
        return tree
    
    if len(tree.subs) > 2:
        new_label = tree.label
        if new_label[-1] != "'":
            new_label += "'"

        new_tree  =  Tree(label = new_label, span = tree.span, subs = tree.subs[1:])
        tree.subs = [tree.subs[0], new_tree]

    for child in tree.subs:
        binarize(child)
    return tree

def main():
    for line in sys.stdin:
        line = line.strip()
        tree = Tree.parse(line)
        tree = binarize(tree)
        print tree

if __name__ == "__main__":
    main()