import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from inception_preprocessing import distort_color, apply_with_random_selector

def random_flip_image_and_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
    The function performs random flip of image and annotation tensors with probability of 1/2
    The flip is performed or not performed for image and annotation consistently, so that
    annotation matches the image.

    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with annotation

    Returns
    -------
    randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
        Randomly flipped image tensor
    randomly_flipped_annotation : Tensor of size (width, height, 1)
        Randomly flipped annotation tensor

    """
    original_shape = tf.shape(annotation_tensor)
    # ensure the annotation tensor has shape (width, height, 1)
    annotation_tensor = tf.cond(tf.rank(annotation_tensor) < 3,
                                lambda: tf.expand_dims(annotation_tensor, axis=2),
                                lambda: annotation_tensor)

    # Random variable: two possible outcomes (0 or 1)
    # with a 1 in 2 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


    randomly_flipped_img = tf.cond(pred=tf.equal(random_var, 0),
                                                 true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                                 false_fn=lambda: image_tensor)

    randomly_flipped_annotation = tf.cond(pred=tf.equal(random_var, 0),
                                                        true_fn=lambda: tf.image.flip_left_right(annotation_tensor),
                                                        false_fn=lambda: annotation_tensor)

    return randomly_flipped_img, tf.reshape(randomly_flipped_annotation, original_shape)


def distort_randomly_image_color(image_tensor, annotation_tensor, fast_mode=False):
    """Accepts image tensor of (width, height, 3) and returns color distorted image.
    The function performs random brightness, saturation, hue, contrast change as it is performed
    for inception model train in TF-Slim (you can find the link below in comments). All the
    parameters of random variables were originally preserved. There are two regimes for the function
    to work: fast and slow. Slow one performs only saturation and brightness random change is performed.

    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3) of tf.int32 or tf.float
        Tensor with image with range [0,255]
    annotation_tensor: The annotation_tensor goes and leaves the function unchanged
    fast_mode : boolean
        Boolean value representing whether to use fast or slow mode

    Returns
    -------
    img_float_distorted_original_range : Tensor of size (width, height, 3) of type tf.float.
        Image Tensor with distorted color in [0,255] intensity range
    """

    # Make the range to be in [0,1]
    img_float_zero_one_range = tf.to_float(image_tensor) / 255

    # Randomly distort the color of image. There are 4 ways to do it.
    # Credit: TF-Slim
    # https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py#L224
    # Most probably the inception models were trainined using this color augmentation:
    # https://github.com/tensorflow/models/tree/master/slim#pre-trained-models
    distorted_image = apply_with_random_selector(img_float_zero_one_range,
                                                 lambda x, ordering: distort_color(x, ordering, fast_mode=fast_mode),
                                                 num_cases=4)

    img_float_distorted_original_range = distorted_image * 255

    return img_float_distorted_original_range, annotation_tensor
