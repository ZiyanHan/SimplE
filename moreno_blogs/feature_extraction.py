# -*- coding: utf-8 -*-
"""
Created on 2020/1/1 21:58
@author: 三剑客
"""

import numpy as np


def train_val_test_split(data):
	val_ratio = test_ratio = 0.1
	size = len(data)
	val_size = int(size * val_ratio)
	test_size = int(size * test_ratio)
	train_size = size - val_size - test_size
	ones_arr = np.ones((size, 1), dtype=np.int32)
	# print(data)
	data2 = np.concatenate([data[:,0:1], ones_arr, data[:, 1:]], axis=1)
	# print(data2.shape, data2)
	np.savetxt("train.txt", data2[:train_size], delimiter="\t", fmt="%d")
	np.savetxt("valid.txt", data2[train_size:train_size+val_size], delimiter="\t",fmt="%d")
	np.savetxt("test.txt", data2[-test_size:], delimiter="\t",fmt="%d")


def main():
	data = np.loadtxt("out.moreno_blogs_blogs", skiprows=2, dtype=np.int32)
	np.random.seed(23)
	np.random.shuffle(data)
	train_val_test_split(data)

if __name__ == '__main__':
    main()