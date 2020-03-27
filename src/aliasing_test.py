import dataset
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.optimize as opt
import tensorflow as tf
import tensorflow.keras.backend as K

def ffttranslate(u,t):
	xdim=1;
	uhat = fft(u,axis=xdim);

	N = u.shape[xdim];
	U  = ifftshift(np.arange(- (N//2),N - ( N//2) ))/N
	U = np.repeat(U[...,np.newaxis], 3, axis=-1)
	ut = np.real(ifft(uhat*np.exp(2j*np.pi * U*t ), axis=xdim));
	return ut
def down_sampling(img):
	""" 
	This function downsamples an image to the shape 30 x 30, with test images at 120 x 120, 
	this is a four times downsampling using a bicubic mode.

	Args :
		img (np.array): image to downsample

	Output :
		img_d (np.array): downsampled image
	"""
	img_d = np.copy(img)
	w,h,_ = img.shape
	img_d = cv2.resize(img_d,(int(h/4), int(w/4)), interpolation = cv2.INTER_CUBIC)
	return img_d
def test_random(model, name_fig1, name_fig2):
	_, _, test_ground_truth, test_data = dataset.images_paths()

	score = [0]
	score_tr = [0]
	for ind in range(len(test_data)):
		img2 = cv2.resize(cv2.imread(test_ground_truth[ind]), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)/255.
		h,w,_ = img2.shape
		img1 = cv2.resize(cv2.imread(test_data[ind]), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)/255.
		for i in range(int((h - 41)/41)):
			for j in range(int((w - 41)/41)):
				t = np.random.uniform()
				c_x = 20 + i * 41
				c_y = 20 + j * 41
				img_test = img1[c_x - 20 : c_x + 21, c_y - 20 : c_y + 21,:]
				img_ground_truth = img2[c_x - 20 : c_x + 21, c_y - 20 : c_y + 21,:]
				prediction = model([img_test])
				score.append(np.mean((prediction - img_ground_truth)**2))
				img_test, img_ground_truth = testing_aliasing(img_test, img_ground_truth, t)
				prediction = model([img_test])
				score_tr.append(np.mean((prediction - img_ground_truth)**2))
		if ind%25 == 0:
			print("testing %i/%i : score - %f ; score tr - %f" %(ind, len(test_data), score[-1], score_tr[-1]))
	plt.plot(score, label = "err without translation")
	plt.plot(score_tr, label = "err with translation")
	plt.legend(loc="upper left")
	plt.xlabel('testing sample')
	plt.ylabel('cumulative error')
	plt.title('evolution of the testing cumulative error\n with and without horizontal translation')
	plt.savefig('figs/random/' + name_fig1, bbox_inches = 'tight')
	plt.close()
	plt.plot(np.array(score_tr) - np.array(score))
	plt.xlabel('testing sample')
	plt.ylabel('difference of the cumulative error')
	plt.title('evolution of the difference of testing cumulative error\n with and without horizontal translation')
	plt.savefig('figs/random/' + name_fig2, bbox_inches = 'tight')
	plt.close()
	model(None, True)
def test_fixed(model, name_fig1, name_fig2, translat):
	_, _, test_ground_truth, test_data = dataset.images_paths()

	score = []
	score_tr = []
	score_tr_tr = []
	score_mutuel = []
	score_mutuel2 = []
	score_tau = []
	best_tau = []
	for ind in range(len(test_data)):
		img = np.array(Image.open(test_ground_truth[ind])).astype(float)
		h,w,_ = img.shape
		for i in range(int((h - 41)/41)):
			for j in range(int((w - 41)/41)):
				c_x = 20 + i * 41
				c_y = 20 + j * 41
				img_ground_truth = img[c_x - 20 : c_x + 21, c_y - 20 : c_y + 21,:]
				img_test = down_sampling(img_ground_truth)
				prediction = 255. * model([img_test])
				score.append(np.mean((prediction - img_ground_truth)**2))
				img_ground_truth_t = ffttranslate(img_ground_truth, translat)
				img_test_t = down_sampling(img_ground_truth_t)
				prediction2 =  255. *  model([img_test_t])
				score_tr.append(np.mean((prediction2 - img_ground_truth_t)**2))
				score_mutuel.append(np.sqrt(np.mean((prediction2 - prediction)**2)))
				prediction_tau = ffttranslate(prediction2, -translat)
				score_mutuel2.append(np.sqrt(np.mean((prediction2 - prediction_tau)**2)))
				score_tau.append(np.sqrt(np.mean((prediction_tau - prediction)**2)))
				score_tr_tr.append(np.sqrt(np.mean((prediction_tau - img_ground_truth)**2)))

				func = lambda x :  np.mean((ffttranslate(prediction2, -x) - prediction)**2)
				tau_opt = opt.minimize(func, translat)
				best_tau.append(tau_opt.x)

		if ind%25 == 0:
			print("testing %i/%i : score - %f ; score tr - %f" %(ind, len(test_data), score[-1], score_tr[-1]))
	# plt.plot(score, label = "err without translation")
	# plt.plot(score_tr, label = "err with translation")
	# plt.legend(loc="upper left")
	# plt.xlabel('testing sample')
	# plt.ylabel('cumulative error')
	# plt.title('evolution of the testing cumulative error\n with and without horizontal translation')
	# plt.savefig('figs/deterministic/' + str(translat) + '_' + name_fig1, bbox_inches = 'tight')
	# plt.close()
	# plt.plot(np.array(score_tr) - np.array(score))
	# plt.xlabel('testing sample')
	# plt.ylabel('difference of the cumulative error')
	# plt.title('evolution of the difference of testing cumulative error\n with and without horizontal translation')
	# plt.savefig('figs/deterministic/' + str(translat) + '_' + name_fig2, bbox_inches = 'tight')
	# plt.close()
	model(None, True)
	return np.mean(score), np.mean(score_tr), np.mean(score_tr_tr), np.mean(score_mutuel), np.mean(score_mutuel2), np.mean(score_tau), np.mean(best_tau), np.std(best_tau)
def plot_score_evol_translat(model, score, score_tr, translat_train):
	if translat_train : 
		model = model + '_translat'
	plt.plot(score, label = "err without translation")
	plt.plot(score_tr, label = "err with translation")
	plt.legend(loc="upper left")
	plt.xlabel('sub-pixel translation')
	plt.ylabel('average error')
	plt.title('evolution of the test error for sub-pixel translation\n model : ' + model)
	plt.savefig('figs/deterministic/' + model + '.png', bbox_inches = 'tight')
	plt.close()
