import sys
from collections import defaultdict 

def find_all_words(line):
    words = []
    w = ""
    for char in line:
        if ord(char) >= ord('A') and ord(char) <= ord('z') or char in ['.', '\'']:
            w += char
        if char == " ":
            w = ""
        if char == ")" and w != "":
            words.append(w)
            w = ""
            
    return words
def main():
    count = defaultdict(int)
    lines = []
    one_count = []
    for line in open("train.trees", "r").readlines():
        line = line.strip()
        lines.append(line)
        
        words = find_all_words(line)
        
        for word in words:
            count[word] += 1
    for word, num in count.items():
        if num > 1:
            sys.stderr.write( word + "\n")
        else:
            one_count.append(word) 
#     print(one_count)
    for line in lines:
#         print(line)
        for word in one_count:
            line = line.replace(' ' + word + ')', " <unk>)")
        print(line)
            
if __name__ == "__main__":
    main()