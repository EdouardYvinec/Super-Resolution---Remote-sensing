import tensorflow as tf
from tensorflow.python.framework import ops

D = 10

def forward_propagation(x, istraining, reuse=tf.AUTO_REUSE):
	with tf.variable_scope('st', reuse=reuse):
		list_kernels_descent = [
			[3,3],
			[3,3],
			[3,3],
			[3,3],
		]
		list_filters_descent = [
			4,
			4,
			8,
			8,
		]
		list_strides_descent = [
			[1,1],
			[1,1],
			[1,1],
			[1,1],
		]
		list_kernels_ascent = [
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[1,1]
		]
		list_filters_ascent = [
			16,
			16,
			8,
			8,
			4,
			4,
			3
		]
		liste_sizes_montee = [
			[16, 16],
			[16, 16],
			[16, 16],
			[32, 32],
			[32, 32],
			[41, 41],
			[41, 41]
		]
		conv_descente = x
		for indice in range(len(list_strides_descent)):
			conv_descente = tf.layers.conv2d(
				inputs = conv_descente,
				filters = list_filters_descent[indice],
				kernel_size = list_kernels_descent[indice],
				strides = list_strides_descent[indice],
				padding = "same",
				activation = None)
			conv_descente = tf.contrib.layers.batch_norm(
				conv_descente, 
				scale=True, 
				is_training=istraining,
				variables_collections=["batch_norm_non_trainable_variables_collection"], 
				activation_fn=tf.nn.relu)

		conv_ascent = conv_descente

		for indice in range(len(list_kernels_ascent)):
			conv_ascent = tf.image.resize_images(
				images = conv_ascent, 
				size = liste_sizes_montee[indice],
				method = tf.image.ResizeMethod.BICUBIC)
			conv_ascent = tf.layers.conv2d(
				inputs = conv_ascent,
				filters = list_filters_ascent[indice],
				kernel_size = list_kernels_ascent[indice],
				strides = [1,1],
				padding = "same",
				activation = None)
			conv_ascent = tf.contrib.layers.batch_norm(
				conv_ascent, 
				scale=True, 
				is_training=istraining,
				variables_collections=["batch_norm_non_trainable_variables_collection"], 
				activation_fn=tf.nn.tanh)
		conv_ascent = tf.layers.conv2d(
			inputs = conv_ascent,
			filters = list_filters_ascent[-1],
			kernel_size = list_kernels_ascent[-1],
			strides = [1,1],
			padding = "same",
			activation = None)
		output = conv_ascent
		output = (output + 1)/2
		return output
def forward_propagation_full_ReLU(x, istraining, reuse=tf.AUTO_REUSE):
	with tf.variable_scope('st', reuse=reuse):
		list_kernels_descent = [
			[3,3],
			[3,3],
			[3,3],
			[3,3],
		]
		list_filters_descent = [
			4,
			4,
			8,
			8,
		]
		list_strides_descent = [
			[1,1],
			[1,1],
			[1,1],
			[1,1],
		]
		list_kernels_ascent = [
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[3,3],
			[1,1]
		]
		list_filters_ascent = [
			16,
			16,
			8,
			8,
			4,
			4,
			3
		]
		liste_sizes_montee = [
			[16, 16],
			[16, 16],
			[16, 16],
			[32, 32],
			[32, 32],
			[41, 41],
			[41, 41]
		]
		conv_descente = x
		for indice in range(len(list_strides_descent)):
			conv_descente = tf.layers.conv2d(
				inputs = conv_descente,
				filters = list_filters_descent[indice],
				kernel_size = list_kernels_descent[indice],
				strides = list_strides_descent[indice],
				padding = "same",
				activation = None)
			conv_descente = tf.contrib.layers.batch_norm(
				conv_descente, 
				scale=True, 
				is_training=istraining,
				variables_collections=["batch_norm_non_trainable_variables_collection"], 
				activation_fn=tf.nn.relu)

		conv_ascent = conv_descente

		for indice in range(len(list_kernels_ascent)):
			conv_ascent = tf.image.resize_images(
				images = conv_ascent, 
				size = liste_sizes_montee[indice],
				method = tf.image.ResizeMethod.BICUBIC)
			conv_ascent = tf.layers.conv2d(
				inputs = conv_ascent,
				filters = list_filters_ascent[indice],
				kernel_size = list_kernels_ascent[indice],
				strides = [1,1],
				padding = "same",
				activation = None)
			conv_ascent = tf.contrib.layers.batch_norm(
				conv_ascent, 
				scale=True, 
				is_training=istraining,
				variables_collections=["batch_norm_non_trainable_variables_collection"], 
				activation_fn=tf.nn.relu)
		conv_ascent = tf.layers.conv2d(
			inputs = conv_ascent,
			filters = list_filters_ascent[-1],
			kernel_size = list_kernels_ascent[-1],
			strides = [1,1],
			padding = "same",
			activation = None)
		output = conv_ascent
		return output
def forward_propagation_from_article(x, istraining, reuse=tf.AUTO_REUSE):
	with tf.variable_scope('st', reuse=reuse):
		# embedding network
		x = tf.layers.conv2d(
			inputs = x,
			filters = 4,
			kernel_size = (3,3),
			strides = (1,1),
			padding = "same",
			activation = tf.nn.relu)
		x = tf.layers.conv2d(
			inputs = x,
			filters = 4,
			kernel_size = (1,1),
			strides = (1,1),
			padding = "same",
			activation = tf.nn.relu)

		# recursive network
		concat_all_partial_outputs = x

		for t in range(D):
			x = tf.layers.conv2d(
				inputs = x,
				filters = 4,
				kernel_size = (3,3),
				strides = (1,1),
				padding = "same",
				activation = tf.nn.relu, 
				name = "recursive_conv")
			concat_all_partial_outputs = tf.concat([concat_all_partial_outputs, x], axis = -1)

		# reconstruction network
		reconstruction = tf.image.resize_images(
			images = concat_all_partial_outputs, 
			size = [41,41],
			method = tf.image.ResizeMethod.BICUBIC)

		reconstruction = tf.layers.conv2d(
			inputs = reconstruction,
			filters = D*3,
			kernel_size = (3,3),
			strides = (1,1),
			padding = "same",
			activation = tf.nn.relu)
		output = tf.layers.conv2d(
			inputs = reconstruction,
			filters = 3,
			kernel_size = (1,1),
			strides = (1,1),
			padding = "same",
			activation = None)
		return output, reconstruction


















