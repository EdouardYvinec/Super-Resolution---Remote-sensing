import tensorflow as tf
from tensorflow.python.framework import ops
import dataset
import cv2
import numpy as np
import random
from aliasing_test import ffttranslate, down_sampling
from kim import forward_propagation, forward_propagation_full_ReLU, forward_propagation_from_article
import matplotlib.pyplot as plt

forward_props = {
	'baseline' : forward_propagation,
	'baseline_tanh' : forward_propagation_full_ReLU,
	'article' : forward_propagation_from_article
}

train_ground_truth, test_ground_truth = dataset.images_paths()
prop_val = int(len(train_ground_truth)/10.)
val_ground_truth = train_ground_truth[:prop_val]
train_ground_truth = train_ground_truth[prop_val:]

def get_minibatch_with_translat(batch, train = True, test = False):
	data = []
	ground_truth = []
	for ind in batch:
		if train:
			img2 = cv2.imread(train_ground_truth[ind])/255.
		elif not test:
			img2 = cv2.imread(val_ground_truth[ind])/255.
		else:
			img2 = cv2.imread(test_ground_truth[ind])/255.
		w,h,_ = img2.shape
		p = np.random.uniform()
		if p < 0.5:
			t = np.random.uniform()
			img2 = ffttranslate(img2, t)
		c_x = 20 + int((w - 41) * np.random.uniform())
		c_y = 20 + int((h - 41) * np.random.uniform())
		img2 = img2[c_x - 20 : c_x + 21, c_y - 20 : c_y + 21,:]
		img1 = down_sampling(img2)
		data.append(img1)
		ground_truth.append(img2)

	return data, ground_truth

FRACTION_GPU_POUR_TF = 0.2

def minibatches(data_set, batch_size = 16):
	n = len(data_set)
	data_indeces = np.arange(len(data_set))
	random.shuffle(data_indeces)
	minibatches = []
	for indice in range(int(n/batch_size)):
		minibatches.append(data_indeces[indice*batch_size : (indice + 1)*batch_size])
	if (indice+1)*batch_size != n:
		minibatches.append(data_indeces[(indice+1)*batch_size:])
	return minibatches
def create_placeholders():
	X = tf.placeholder(
		tf.float32, 
		shape = (None,10,10,3), 
		name = 'X')
	Y = tf.placeholder(
		tf.float32, 
		shape = (None,41,41,3), 
		name = 'Y')
	return X,Y
def cost_superRes(Z, Y):
	cost = tf.reduce_mean(tf.pow(Z - Y,2), axis = (-1,-2,-3))
	return cost
def cost_out_bounds(Z):
	cost_out_bounds = tf.reduce_sum(tf.nn.relu(-Z), axis = (-1,-2,-3)) + tf.reduce_sum(tf.nn.relu(Z - 1), axis = (-1,-2,-3))
	return cost_out_bounds
def cost_article(Z, Rec, Y):
	cost_final_output = tf.reduce_mean(tf.pow(Z - Y,2), axis = (-1,-2,-3))
	Y_Rec = Y
	for i in range(9):
		Y_Rec = tf.concat([Y_Rec, Y], axis = -1)
	cost_partial_output = tf.reduce_mean(tf.pow(Rec - Y_Rec,2), axis = (-1,-2,-3))/10.
	return cost_final_output + cost_partial_output
def model(model_type, learning_rate = 0.02, num_epochs = 10, minibatch_size = 64):
	best_val = np.inf
	ops.reset_default_graph()
	costs = []
	val_costs = []
	X, Y = create_placeholders()
	TF_ISTRAINING_PARAM = tf.placeholder(dtype=tf.bool, shape=())
	num_steps = int(num_epochs * (int(len(train_ground_truth)/minibatch_size) + 1))
	curr_step = tf.placeholder(dtype=tf.float32, shape=())
	learning_rate2 = tf.scalar_mul(learning_rate, tf.pow((1 - curr_step / num_steps), 0.9))
	if model_type == 'article':
		Z, Rec = forward_props[model_type](X, TF_ISTRAINING_PARAM)
		cost1 = cost_article(Z, Rec, Y)
		cost2 = cost1 * 0
	else:
		Z = forward_props[model_type](X, TF_ISTRAINING_PARAM)
		cost1 = cost_superRes(Z, Y)
		cost2 = cost_out_bounds(Z)
	cost2 = cost_out_bounds(Z)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate2,beta1=0.9).minimize(cost1 + cost2)
	init = tf.initialize_all_variables()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FRACTION_GPU_POUR_TF)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			mini_batches = minibatches(train_ground_truth,minibatch_size)
			num_minibatches = len(mini_batches)
			for cpt in range(num_minibatches):
				minibatch_X, minibatch_Y = get_minibatch_with_translat(mini_batches[cpt])
				_ , temp_cost1, temp_cost2 = sess.run([optimizer, cost1, cost2], 
					feed_dict={
					X: minibatch_X,
					Y: minibatch_Y,
					curr_step: cpt + epoch * num_minibatches,
					TF_ISTRAINING_PARAM: True
					})
				temp_cost1 = np.mean(temp_cost1)
				temp_cost2 = np.mean(temp_cost2)
				costs.append(temp_cost1)
				print(model_type + " [translat] - epoch : %i/%i, batch : %i/%i, current training error : %f (res) - %.2f (bounds)"
					%(epoch, num_epochs, cpt, num_minibatches, temp_cost1, temp_cost2))

			mini_batches = minibatches(val_data,minibatch_size)
			num_minibatches = len(mini_batches)
			val_cost = 0.
			for cpt in range(num_minibatches):
				minibatch_X,minibatch_Y = get_minibatch_with_translat(mini_batches[cpt], train = False)
				temp_cost1, temp_cost2 = sess.run([cost1, cost2], 
					feed_dict={
					X: minibatch_X,
					Y: minibatch_Y,
					curr_step: cpt + epoch * num_minibatches,
					TF_ISTRAINING_PARAM: False
					})
				temp_cost1 = np.mean(temp_cost1)
				temp_cost2 = np.mean(temp_cost2)
				val_cost += temp_cost1
				print("epoch : %i/%i, batch : %i/%i, current validation error : %f"
					%(epoch, num_epochs, cpt, num_minibatches, temp_cost1))
			val_costs.append(val_cost)
			if epoch > 0 and val_cost < best_val:
				best_val = val_cost
				all_vars = tf.global_variables()
				saver = tf.train.Saver(var_list=all_vars)
				saver.save(sess,  "../checkpoint_model/superRes_" + model_type + "_translat.ckpt")
	plt.plot(costs)
	plt.xlabel('steps')
	plt.ylabel('error')
	plt.title('cout en train ' + model_type + ' [avec translat]')
	plt.savefig('cout en train ' + model_type + ' translat.png', bbox_inches = 'tight')
	plt.close()
	plt.plot(val_costs)
	plt.xlabel('steps')
	plt.ylabel('error')
	plt.title('cout en val ' + model_type + ' [avec translat]')
	plt.savefig('cout en val ' + model_type + ' translat.png', bbox_inches = 'tight')
	plt.close()
def reload(model_type):
	ops.reset_default_graph()
	costs = []
	X, Y = create_placeholders()
	TF_ISTRAINING_PARAM = tf.placeholder(dtype=tf.bool, shape=())
	if model_type == 'article':
		Z, Rec = forward_props[model_type](X, TF_ISTRAINING_PARAM)
	else:
		Z = forward_props[model_type](X, TF_ISTRAINING_PARAM)

	all_vars = tf.all_variables()
	save_vars = [k for k in all_vars if k.name.startswith("st")]
	saver = tf.train.Saver(var_list=save_vars)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FRACTION_GPU_POUR_TF)

	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	saver.restore(sess, "../checkpoint_model/superRes_" + model_type + "_translat.ckpt")

	def model(x, close = False):
		if close:
			sess.close()
			return None
		prediction = sess.run([Z], 
			feed_dict={
			X: x,
			TF_ISTRAINING_PARAM: False
		})[0]
		return prediction
	return model


