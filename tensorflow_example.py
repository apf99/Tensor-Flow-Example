import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_features = 28 * 28
n_classes = 10
batch_size = 100
learning_rate = 0.01

###################Configuration########################## 
# List containing the # of nodes in each layer
n_nodes = [n_features, 2500, 2000, 1500, 1000, 500, n_classes]
n_epochs = 30
########################################################## 

n_hidden_layers = len(n_nodes) - 2

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float')


def neural_network_model(data):
	# define the layers
	layers = [] 
	for i in range(n_hidden_layers + 1):
		layers.append( {'weights':tf.Variable(tf.random_normal([n_nodes[i], n_nodes[i+1]])), 'biases':tf.Variable(tf.random_normal([n_nodes[i+1]]))} )

	# calculate the nodal values for each layer
	calcs = [data]
	for i in range(n_hidden_layers):
		calcs.append( tf.nn.relu(tf.matmul(calcs[i], layers[i]['weights']) + layers[i]['biases']) )

	#  return the last layer of nodes
	return tf.matmul(calcs[-1], layers[-1]['weights']) + layers[-1]['biases']

	
def train_neural_network(x, learning_rate):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range( int(mnist.train.num_examples / batch_size) ):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print 'Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
		print 'Hidden Layers:', n_nodes[1:-1]
		print 'Learning Rate:', learning_rate


train_neural_network(x, learning_rate)