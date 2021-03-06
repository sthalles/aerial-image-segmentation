import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from read_data import get_labels_from_annotation_batch

def model_input(input_image_shape, label_image_shape):
    # Create and return the placeholders for receiving data
    batch_images_placeholder = tf.placeholder(tf.float32, shape=[None] + input_image_shape)
    batch_labels_placeholder = tf.placeholder(tf.uint8, shape=[None] + label_image_shape)
    is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")
    keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")
    return batch_images_placeholder, batch_labels_placeholder, is_training_placeholder, keep_prob

@slim.add_arg_scope
def __block_unit(net, k, id, keep_prob=1.0, rate=1):
    # block_unit implements a bottleneck layer
    with tf.variable_scope("block_unit_" + str(id)):

        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        # each 1×1 convolution produce 4k feature-maps
        net = slim.conv2d(net, 4*k, [1,1], scope="conv1", activation_fn=None, normalizer_fn=None)
        net = slim.dropout(net, keep_prob, scope='dropout1')
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        net = slim.conv2d(net, k, [3,3], scope="conv3x3", activation_fn=None, normalizer_fn=None, rate=rate)
        net = slim.dropout(net, keep_prob, scope='dropout2')

        return net

@slim.add_arg_scope
def add_dense_block(input_tensor, k, number_of_units, scope, multi_grid=[], rate=1, keep_prob=1.0, summary=True):

    with tf.variable_scope(scope):
        block = input_tensor
        concat_op_name = "concat_1"
        for i in range(0,number_of_units):

            # compute atrous conv rate
            if not multi_grid:
                if len(multi_grid) != number_of_units:
                    raise("The length of the multi_grid array must match the number of units.")
                rate = rate*multi_grid[i]

            unit = __block_unit(block, k, i, keep_prob, rate=rate)
            concat_op_name += "_" + str(i)

            if i == 0:
                block = unit
            else:
                block = tf.concat((block, unit), axis=3, name=concat_op_name)
        if summary:
            tf.summary.image(scope, block[:,:,:,:1], 1)
        return block


@slim.add_arg_scope
def add_transition_layer(net, keep_prob, scope, pooling=True, theta=1.0):
    # We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks
    with tf.variable_scope(scope):
        current_depth = slim.utils.last_dimension(net.get_shape(), min_rank=4)

        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, theta*current_depth, [1,1], scope='conv1x1',
                          activation_fn=None, normalizer_fn=None)
        net = slim.dropout(net, keep_prob, scope='dropout')
        if pooling:
            net = slim.avg_pool2d(net, [2,2], scope='avg_pool', stride=2)
        return net


@slim.add_arg_scope
def add_upsampling_layer(net, size, scope, theta=0.5):

    with tf.variable_scope(scope):
        depth = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        new_depth = int(theta*depth)
        net = slim.conv2d(net, new_depth, [1,1], scope='conv1x1',
                          activation_fn=None, normalizer_fn=None)
        net = tf.image.resize_nearest_neighbor(net, size)
        net = slim.conv2d(net, new_depth, [3,3], scope='conv3x3',
                          activation_fn=None, normalizer_fn=None)
        return net

@slim.add_arg_scope
def add_aspp_layer(net, scope, summary=True):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: network tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :param summary: output tensorboard summary statistics
    :return: network layer with aspp applyed to it.
    """
    with tf.variable_scope(scope):
        depth = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        at_pool1x1 = slim.conv2d(net, depth, [1,1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_1", rate=6, activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_2", rate=12, activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_3", rate=18, activation_fn=None)

        net = tf.concat((net, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")
        net = slim.conv2d(net, depth, [1,1], scope="conv_1x1_output", activation_fn=None)
        if summary:
            tf.summary.image("net", net[:,:,:,:1], 1)
        return net


def model(batch_images, args):
    """

    :param batch_images: tensor of shape [batch_size, height, width, channels]
    :return: Tensor of the same shape as [batch_images]
    """

    batch_norm_params = {
      'decay': args.batch_norm_decay,
      'epsilon': args.batch_norm_epsilon,
      'scale': True,
      'is_training': args.is_training,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    mask_np = np.ones((args.crop_size,args.crop_size,3), dtype =np.float32)
    mask_np[:,:,0] = 0
    mask_tensor = tf.constant(mask_np)

    # batch_images shape (?, 224,224,3)
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        stride=1,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(args.l2_regularizer)):

        with slim.arg_scope([slim.batch_norm], **batch_norm_params):

            # We refer to layers between blocks as transition layers, which do convolution and pooling
            net = slim.conv2d(batch_images, 2*args.growth_rate, [7,7], scope='conv1', stride=2) # output stride 2

            # dense block 1
            block_1 = add_dense_block(net, args.growth_rate, 6, "block_1", args.keep_prob)

            net = add_transition_layer(block_1, args.keep_prob, scope="transition_layer_1") # output stride 4

            # dense block 2
            block_2 = add_dense_block(net, args.growth_rate, 12, "block_2", args.keep_prob)

            net = add_transition_layer(block_2, args.keep_prob, scope="transition_layer_2", pooling=False) # output stride 4

            # dense block 3
            block_3 = add_dense_block(net, args.growth_rate, 24, "block_3", args.keep_prob)

            net = add_transition_layer(block_3, args.keep_prob, scope="transition_layer_3", pooling=False) # output stride 8

            #####################################
            # DenseNet based Decoder
            #####################################

            block_2_shape = tf.shape(block_2)
            new_size = (block_2_shape[1], block_2_shape[2])
            net = add_upsampling_layer(net, new_size, scope="upsampling_layer_1")  # output stride 4

            multi_grid = [1,2,1]
            rate = 2
            net = add_dense_block(net, args.growth_rate, 3, scope="block_4",
                                  keep_prob=args.keep_prob, multi_grid=multi_grid, rate=rate)

            # if true, implement an Atrous Spatial Pyrimid Pooling layer before the logits output.
            if args.aspp:
                net = add_aspp_layer(net, "aspp_layer_1")

            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, args.number_of_classes, (3,3), activation_fn=None, normalizer_fn=None)

            # resize the image to its original size
            batch_images_shape = tf.shape(batch_images)
            new_size = (batch_images_shape[1], batch_images_shape[2])

            if args.image_summary:
                net = tf.image.resize_images(net, new_size)

            # If true, zero out all the logits from class 0 before calculating the softmax with cross entropy loss
            if args.zero_class_logits:
                net = tf.cond(args.is_training, lambda: tf.multiply(net, mask_tensor), lambda: net)

            tf.summary.image("output", net, 1)
            return net

def _compute_cross_entropy_mean(class_weights, labels, softmax):
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), class_weights),
                                   reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    return cross_entropy_mean


def model_loss(logits, labels, number_of_classes, class_weights=[0.9,1.0,1.01], epsilon=1e-10):
    """
    Perform softmax with cross entropy with individual class weights
    :param logits: network output tensor of shape [batch_size, height, width, channels]
    :param labels: ground truth tensor of shape [batch_size, height, width, channels]
    :param number_of_classes: number of classes to be predicted
    :param class_weights: weight values to reweight the cross entropy loss
    :param epsilon: Because there might be batches that do not have all the classes in it, we add an small value to make the cross entropy viable
    :return:
    """
    "Returns the cross entropy mean to be minimized, the predictions for each pixel and the probabilities."
    epsilon = tf.constant(value=epsilon)

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    labels = tf.reshape(tf.one_hot(tf.cast(label_flat, tf.uint8), depth=number_of_classes), (-1, number_of_classes))

    probabilities = tf.nn.softmax(logits) + epsilon

    # αc = median freq/freq(c)
    cross_entropy_mean = _compute_cross_entropy_mean(class_weights,labels, tf.reshape(probabilities,(-1, number_of_classes)))

    # Add summary op for the loss -- to be able to see it in tensorboard.
    tf.summary.scalar('cross_entropy_loss', cross_entropy_mean)

    # Tensor to get the final prediction for each pixel -- pay
    # attention that we don't need softmax in this case because
    # we only need the final decision. If we also need the respective
    # probabilities we will have to apply softmax.
    pred = tf.argmax(logits, dimension=3)
    return cross_entropy_mean, pred, probabilities


def model_loss(logits, labels, class_labels):
    "Returns the cross entropy mean to be minimized, the predictions for each pixel and the probabilities."
    annotation_mask = get_labels_from_annotation_batch(labels, class_labels)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=annotation_mask)

    cross_entropy_mean = tf.reduce_mean(cross_entropies)

    # Add summary op for the loss -- to be able to see it in tensorboard.
    tf.summary.scalar('cross_entropy_loss', cross_entropy_mean)

    # Tensor to get the final prediction for each pixel -- pay
    # attention that we don't need softmax in this case because
    # we only need the final decision. If we also need the respective
    # probabilities we will have to apply softmax.
    pred = tf.argmax(logits, dimension=3)
    probabilities = tf.nn.softmax(logits)

    return cross_entropy_mean, pred, probabilities