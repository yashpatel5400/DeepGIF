import tensorflow as tf
from model.common import conv2d, bias_variable, weight_variable, max_pool

def default_N4():
    params = {
        'm1': 48,
        'm2': 48,
        'm3': 48,
        'm4': 48,
        'fc': 200,
        'lr': 0.001,
        'out': 101
    }
    return N4(params)


class N4:
    def __init__(self, params):

        # Hyperparameters
        map_1 = params['m1']
        map_2 = params['m2']
        map_3 = params['m3']
        map_4 = params['m4']
        fc = params['fc']
        learning_rate = params['lr']
        self.out = params['out']
        self.fov = 95
        self.inpt = self.fov + 2 * (self.out // 2)

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        # layer 1 - original stride 1
        W_conv1 = weight_variable([4, 4, 1, map_1])
        b_conv1 = bias_variable([map_1])
        h_conv1 = tf.nn.relu(conv2d(self.image, W_conv1, dilation=1) + b_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, map_1, map_2])
        b_conv2 = bias_variable([map_2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2) + b_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, map_2, map_3])
        b_conv3 = bias_variable([map_3])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4) + b_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=4)

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, map_3, map_4])
        b_conv4 = bias_variable([map_4])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8) + b_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1, 1], dilation=8)

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, map_4, fc])
        b_fc1 = bias_variable([fc])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=16) + b_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, fc, 2])
        b_fc2 = bias_variable([2])
        self.prediction = conv2d(h_fc1, W_fc2, dilation=16) + b_fc2

        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.prediction, self.target))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.loss_summary = tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.binary_prediction = tf.round(self.sigmoid_prediction)
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))
        self.pixel_error_summary = tf.summary.scalar('pixel_error', self.pixel_error)
        
        self.summary_op = tf.summary.merge([
            self.loss_summary,
            self.pixel_error_summary
        ])

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        self.model_name = str.format('out-{}_lr-{}_m1-{}_m2-{}_m3-{}_m4-{}_fc-{}', self.out, learning_rate, map_1,
                                     map_2, map_3, map_4, fc)
