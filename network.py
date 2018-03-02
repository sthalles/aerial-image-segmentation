import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from read_data import get_labels_from_annotation_batch
#from utils import bilinear_upsample_weights
from preprocessing import distort_randomly_image_color
from densenet import densenet
from densenet import densenet_utils
from deeplab import deeplab_v3
from deeplab import deeplab_utils_utils

def model_input():
    # Create and return the placeholders for receiving data
    is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")
    return  is_training_placeholder

@slim.add_arg_scope
def add_upsampling_layer(net, size, scope, theta=1.0):

    with tf.variable_scope(scope):
        depth = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        new_depth = int(theta*depth)
        net = slim.conv2d(net, new_depth, [1,1], scope='conv1x1',
                          activation_fn=None, normalizer_fn=None)

        net = tf.image.resize_bilinear(net, size)
        net = slim.conv2d(net, new_depth, [3,3], scope='conv3x3',
                          activation_fn=None, normalizer_fn=None)
        return net

@slim.add_arg_scope
def add_aspp_layer(net, scope, summary=True, depth=64):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: network tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :param summary: output tensorboard summary statistics
    :return: network layer with aspp applyed to it.
    """
    with tf.variable_scope(scope):
        at_pool1x1 = slim.conv2d(net, depth, [1,1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_1", rate=4, activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_2", rate=8, activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3,3], scope="conv_3x3_3", rate=12, activation_fn=None)

        net = tf.concat((net, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")
        net = slim.conv2d(net, depth, [1,1], scope="conv_1x1_output", activation_fn=None)
        if summary:
            tf.summary.image("net", net[:,:,:,:1], 1)
        return net


@slim.add_arg_scope
def asspp_with_global_avg_pool(net, rates, scope, summary=True, depth=256):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: network tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :param summary: output tensorboard summary statistics
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_global_poolconv_1x1",
                                           activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rates[0], activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=rates[1], activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=rates[2], activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        if summary:
            tf.summary.image("net", net[:, :, :, :1], 1)
        return net


@slim.add_arg_scope
def apply_data_augmentation(batch_images):
    batch_images = tf.map_fn(lambda img: distort_randomly_image_color(img), batch_images)
    return batch_images

@slim.add_arg_scope
def add_skip_connection(layer1, layer2, scope=None):
    with tf.variable_scope(scope):
        layer1_depth_in = slim.utils.last_dimension(layer1.get_shape(), min_rank=4)
        layer2_depth_in = slim.utils.last_dimension(layer2.get_shape(), min_rank=4)

        if layer1_depth_in != layer2_depth_in:
            # downsample layer 1
            layer2 = slim.conv2d(layer2, layer1_depth_in, [1, 1],
                                   normalizer_fn=None, activation_fn=None,scope='shortcut')

        output = layer1 + layer2
        return output



@slim.add_arg_scope
def deeplab_segmentation(batch_images, args, is_training, reuse):
    """
    :param batch_images: tensor of shape [batch_size, height, width, channels]
    :return: Tensor of the same shape as [batch_images]
    """

    with tf.variable_scope("network", reuse=reuse):

        if args.normalizer == "mean_subtraction":
            # Using Mean Subtraction nomalization
            batch_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), batch_images)
        elif args.normalizer == "standard":
            # Zero mean and equal variance normalization
            batch_images = (batch_images - 128.) / 128.
        elif args.normalizer == "simple_norm":
            batch_images = tf.divide(batch_images, 255., name="data_normalization")

        with slim.arg_scope(deeplab_utils_utils.deeplab_arg_scope(args.l2_regularizer, args.batch_norm_decay, args.batch_norm_epsilon)):

            net, end_points = deeplab_v3.deeplab_v3_50(batch_images,
                                                     args.number_of_classes,
                                                     is_training=is_training,
                                                     global_pool=False,
                                                     output_stride=args.output_stride,
                                                     initial_output_stride=2,
                                                     spatial_squeeze=False,
                                                     reuse=reuse)

            # resize the image to its original size
            batch_images_shape = tf.shape(batch_images)
            new_size = (batch_images_shape[1], batch_images_shape[2])
            net = tf.image.resize_bilinear(net, new_size)

            tf.summary.image("output", net, 1)
            return net

@slim.add_arg_scope
def upsample(net, size, scope):
    with tf.variable_scope(scope):
        depth = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        net = tf.image.resize_bilinear(net, size)
        return slim.conv2d(net, depth, [3, 3], scope='conv3x3',
                          activation_fn=None, normalizer_fn=None)


@slim.add_arg_scope
def densenet_fcn_segmentation(batch_images, args, is_training, reuse):
    """
    :param batch_images: tensor of shape [batch_size, height, width, channels]
    :return: Tensor of the same shape as [batch_images]
    """

    with tf.variable_scope("network", reuse=reuse):

        if args.normalizer == "mean_subtraction":
            # Using Mean Subtraction nomalization
            batch_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), batch_images)
        elif args.normalizer == "standard":
            # Zero mean and equal variance normalization
            batch_images = (batch_images - 128.) / 128.
        elif args.normalizer == "simple_norm":
            batch_images = tf.divide(batch_images, 255., name="data_normalization")

        with slim.arg_scope(densenet_utils.densenet_arg_scope(args.l2_regularizer, args.batch_norm_decay, args.batch_norm_epsilon)):

            logits, end_points = densenet.densenet_121(batch_images,
                                         args.number_of_classes,
                                         is_training=is_training,
                                         global_pool=False,
                                         output_stride=args.output_stride, # after chaning the reuse parameter, it is working with output_stride=4
                                         include_root_max_poolling=False,
                                         spatial_squeeze=False,
                                         reuse=reuse)

            block1 = end_points['network/DenseNet_121/block1']

            with slim.arg_scope([slim.batch_norm], is_training=is_training):


                # upsample 2
                block1_shape = tf.shape(block1)
                new_size = (block1_shape[1], block1_shape[2])
                net = upsample(logits, new_size, "simple_upsample_2")

                block1_logits = slim.conv2d(block1,
                                           args.number_of_classes,
                                           [1, 1],
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           weights_initializer=tf.zeros_initializer,
                                           scope='block1_conv1x1')
                net = block1_logits + net

                # upsample 3
                # resize the image to its original size
                batch_images_shape = tf.shape(batch_images)
                new_size = (batch_images_shape[1], batch_images_shape[2])
                net = upsample(net, new_size, "simple_upsample_3")

                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                net = slim.conv2d(net, args.number_of_classes, (3,3), activation_fn=None, normalizer_fn=None)

                # If true, zero out all the logits from class 0 before calculating the softmax with cross entropy loss
                if args.channel_wise_inhibited_softmax:
                    net = tf.multiply(net, [0,1,1], name="channel_wise_inhibited_softmax")

                tf.summary.image("output", net, 1)
                return net


@slim.add_arg_scope
def densenet_segmentation(batch_images, args, is_training, reuse):
    """
    :param batch_images: tensor of shape [batch_size, height, width, channels]
    :return: Tensor of the same shape as [batch_images]
    """

    with tf.variable_scope("network", reuse=reuse):

        if args.normalizer == "mean_subtraction":
            # Using Mean Subtraction nomalization
            batch_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), batch_images)
        elif args.normalizer == "standard":
            # Zero mean and equal variance normalization
            batch_images = (batch_images - 128.) / 128.
        elif args.normalizer == "simple_norm":
            batch_images = tf.divide(batch_images, 255., name="data_normalization")

        with slim.arg_scope(densenet_utils.densenet_arg_scope(args.l2_regularizer, args.batch_norm_decay, args.batch_norm_epsilon)):

            _, end_points = densenet.densenet_121(batch_images,
                                         args.number_of_classes,
                                         is_training=is_training,
                                         global_pool=False,
                                         output_stride=8, # after chaning the reuse parameter, it is working with output_stride=4
                                         include_root_max_poolling=False,
                                         spatial_squeeze=False,
                                         reuse=reuse)

            net = end_points['network/DenseNet_121/block4']

            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                if args.aspp:
                    net = add_aspp_layer(net, "aspp_layer_1")

                # resize the image to its original size
                batch_images_shape = tf.shape(batch_images)
                new_size = (batch_images_shape[1], batch_images_shape[2])
                net = add_upsampling_layer(net, new_size, theta=args.theta, scope="upsampling_layer_1")

                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                net = slim.conv2d(net, args.number_of_classes, (3,3), activation_fn=None, normalizer_fn=None)

                # If true, zero out all the logits from class 0 before calculating the softmax with cross entropy loss
                if args.channel_wise_inhibited_softmax:
                    net = tf.multiply(net, [0,1,1], name="channel_wise_inhibited_softmax")

                tf.summary.image("output", net, 1)
                return net


def _compute_cross_entropy_mean(class_weights, labels, softmax):
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), class_weights),
                                   reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    return cross_entropy_mean


def model_loss_with_class_balancing(logits, labels, number_of_classes, class_weights=None, epsilon=1e-10):
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
    pred = tf.argmax(logits, axis=3)
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
    pred = tf.argmax(logits, axis=3)
    probabilities = tf.nn.softmax(logits)

    return cross_entropy_mean, pred, probabilities
