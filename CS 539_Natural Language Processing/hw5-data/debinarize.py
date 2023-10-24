def deBinarize(t):
    if t.subs is not None:

        while t.subs[-1].label[-1] == "'":
            node, label = t.subs[-1], t.subs[-1].label

            t.subs.pop()
            t.subs.extend(node.subs)

    if t.subs is not None:
        for child in t.subs:
            deBinarize(child)
    return t