import pdb
import sys, itertools
import pickle
from collections import OrderedDict

characters = ['_','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
start_char = '<s>'
end_char = '</s>'

class MakeGrams(object):
	def __init__(self, grams, smoothValue):
		super(MakeGrams, self).__init__()
		self.trans_dict = OrderedDict()
		self.grams = grams
		self.smoothValue = smoothValue

		# Enumerate all possible transitions
		if self.grams > 1:
			for i in range(0, self.grams-1):
				for state_from in itertools.product(characters, repeat=i):
					state_from = start_char + ''.join(state_from)
					for next_char in characters+[end_char]:
						self.add_transition(state_from, next_char)

		for state_from in itertools.product(characters, repeat=self.grams-1):
			state_from = ''.join(state_from)
			for next_char in characters+[end_char]:
				self.add_transition(state_from, next_char)

	def train(self, trainFile):
		file_in = open(trainFile, "r")

		for line in file_in:
			line = list(line.strip().replace(' ', '_'))
			line = ['<s>'] + line + ['</s>']
			for i in range(1, len(line)):
				prev_state = ''
				for j in range(max(i-self.grams+1, 0), i):
					if j < 0:
						break
					prev_state += line[j]

				self.add_transition(prev_state, line[i])

		# Normalize model
		for x in self.trans_dict:
			total = sum(self.trans_dict[x].values())
			for y in self.trans_dict[x]:
				self.trans_dict[x][y] = self.trans_dict[x][y] / total

	def add_transition(self, state_from, next_char):
		state_from = '<s>' if state_from == '' else state_from

		if state_from not in self.trans_dict:
			self.trans_dict[state_from] = dict()

		if next_char not in self.trans_dict[state_from]:
			self.trans_dict[state_from][next_char] = self.smoothValue
		else:
			self.trans_dict[state_from][next_char] += 1

	def get_trans_dict(self):
		return self.trans_dict

