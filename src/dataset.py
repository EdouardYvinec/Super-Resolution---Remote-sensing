import numpy as np
import os

def images_paths():
	train_ground_truth = []
	test_ground_truth = []
	image_names = []

	list_images = os.listdir('../dataset/DIV2K_train_HR/')
	list_images.sort()
	for elem in list_images:
		if elem[0] != '.':
			train_ground_truth.append('../dataset/DIV2K_train_HR/' + elem)
	list_images = os.listdir('../dataset/DIV2K_valid_HR/')
	list_images.sort()
	for elem in list_images:
		test_ground_truth.append('../dataset/DIV2K_valid_HR/' + elem)

	return train_ground_truth, test_ground_truth
