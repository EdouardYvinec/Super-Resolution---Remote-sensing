from model import resolve_single
from model.edsr import edsr
from PIL import Image
import cv2
import dataset
import numpy as np
import scipy.optimize as opt
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

weights_file = "../weights/edsr-16-x4/weights.h5"

def ffttranslate(u,t):
	"""
	This function computes the horizontal translation of an RGB image (input shape should be [w,h,d])

	Args :
		- u (np.array): image on which we apply the translation
		- t (float): sub-pixel translation

	Output :
		- ut (np.array): translated image
	"""
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
def load_pre_trained_model():
	"""
	This function defines the architecture of the CNN and loads the pre-trained weights from :
	https://github.com/krasserm/super-resolution

	Args :
		- None

	Output :
		model (tensorflow/keras): CNN
	"""
	scale = 4
	depth = 16
	model = edsr(scale=scale, num_res_blocks=depth)
	model.load_weights(weights_file)
	return model
def test_sub_pix_translation(translat):
	"""
	To define the tests run by this function we note P the prediction function, the ground truth image G 
	and I the downsampled image from G.
	We note G_t the image the translated image G by t on the horizontal axis and I_t the downsampled image from G_t. 

	sc = || G - P(I) ||
	sc_tr = || G_t - P(I_t) ||
	sc_mu = || P(I_t) - P(I) ||
	sc_tau = || P(I_t)_(-t) - P(I) ||
	best_tau = min || P(I_t)_(t') - P(I) ||

	the minimum is searched over t'.

	Args :
		- translat (float): value of t

	Output :
		- average of the previous scores 
	"""
	eval_model = load_pre_trained_model()
	_, test_ground_truth = dataset.images_paths()
	
	score = []
	score_tr = []
	score_tr_tr = []
	score_mutuel = []
	score_mutuel2 = []
	score_tau = []
	best_tau = []

	for ind in range(len(test_ground_truth)):
		img = np.array(Image.open(test_ground_truth[ind])).astype(float)
		h,w,_ = img.shape
		img_ground_truth = img
		img_test = down_sampling(img_ground_truth)

		prediction = resolve_single(eval_model, img_test)
		prediction = K.eval(prediction).astype(float)
		score.append(np.sqrt(np.mean((prediction - img_ground_truth)**2)))

		img_ground_truth_t = ffttranslate(img_ground_truth, translat)
		img_test_t = down_sampling(img_ground_truth_t)

		prediction2 = resolve_single(eval_model, img_test_t)
		prediction2 = K.eval(prediction2).astype(float)
		score_tr.append(np.sqrt(np.mean((prediction2 - img_ground_truth_t)**2)))

		score_mutuel.append(np.sqrt(np.mean((prediction2 - prediction)**2)))
		prediction_tau = ffttranslate(prediction2, -translat)
		score_mutuel2.append(np.sqrt(np.mean((prediction2 - prediction_tau)**2)))
		score_tau.append(np.sqrt(np.mean((prediction_tau - prediction)**2)))
		score_tr_tr.append(np.sqrt(np.mean((prediction_tau - img_ground_truth)**2)))

		func = lambda x :  np.mean((ffttranslate(prediction2, -x) - prediction)**2)
		tau_opt = opt.minimize(func, translat)
		best_tau.append(tau_opt.x)
		print("t = %f - testing pre-trained model %i/%i : score - %f ; score tr - %f ; score mutuel - %f ; score translation inv - %f ; best tau - %f" 
			%(translat, ind, len(test_ground_truth), score[-1], score_tr[-1], score_mutuel[-1], score_tau[-1], best_tau[-1]))
	return np.mean(score), np.mean(score_tr), np.mean(score_tr_tr), np.mean(score_mutuel), np.mean(score_mutuel2), np.mean(score_tau), np.mean(best_tau), np.std(best_tau)

def plot_some_tests():
	img_address = "../dataset/DIV2K_valid_HR/0892.png"
	eval_model = load_pre_trained_model()
	img = np.array(Image.open(img_address)).astype(float)
	h,w,_ = img.shape
	cv2.imwrite("data.png", img)
	translat = [0,0.25,1]
	for i,t in enumerate(translat):
		img_test = ffttranslate(img, t)
		img_test = down_sampling(img_test)
		pred = K.eval(resolve_single(eval_model, img_test))
		pred2 = cv2.resize(img_test,(w, h), interpolation = cv2.INTER_CUBIC)
		cv2.imwrite("VT_" + str(t) + ".png", img_test)
		cv2.imwrite("Pred_" + str(t) + ".png", pred)
		cv2.imwrite("base_" + str(t) + ".png", pred2)

def test_baseline(translat):
	_, test_ground_truth = dataset.images_paths()
	
	score = []
	score_tr = []
	score_tr_tr = []
	score_mutuel = []
	score_mutuel2 = []
	score_tau = []
	best_tau = []

	for ind in range(len(test_ground_truth)):
		img = np.array(Image.open(test_ground_truth[ind])).astype(float)
		h,w,_ = img.shape
		img_ground_truth = img
		img_test = down_sampling(img_ground_truth)

		prediction = cv2.resize(img_test,(w, h), interpolation = cv2.INTER_CUBIC)
		score.append(np.sqrt(np.mean((prediction - img_ground_truth)**2)))

		img_ground_truth_t = ffttranslate(img_ground_truth, translat)
		img_test_t = down_sampling(img_ground_truth_t)

		prediction2 = cv2.resize(img_test_t,(w, h), interpolation = cv2.INTER_CUBIC)
		score_tr.append(np.sqrt(np.mean((prediction2 - img_ground_truth_t)**2)))

		score_mutuel.append(np.sqrt(np.mean((prediction2 - prediction)**2)))
		prediction_tau = ffttranslate(prediction2, -translat)
		score_mutuel2.append(np.sqrt(np.mean((prediction2 - prediction_tau)**2)))
		score_tau.append(np.sqrt(np.mean((prediction_tau - prediction)**2)))
		score_tr_tr.append(np.sqrt(np.mean((prediction_tau - img_ground_truth)**2)))

		func = lambda x :  np.mean((ffttranslate(prediction2, -x) - prediction)**2)
		tau_opt = opt.minimize(func, translat)
		best_tau.append(tau_opt.x)
		print("t = %f - testing pre-trained model %i/%i : score - %f ; score tr - %f ; score mutuel - %f ; score translation inv - %f ; best tau - %f" 
			%(translat, ind, len(test_ground_truth), score[-1], score_tr[-1], score_mutuel[-1], score_tau[-1], best_tau[-1]))
	return np.mean(score), np.mean(score_tr), np.mean(score_tr_tr), np.mean(score_mutuel), np.mean(score_mutuel2), np.mean(score_tau), np.mean(best_tau), np.std(best_tau)






