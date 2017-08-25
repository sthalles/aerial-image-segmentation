import argparse
from network import model_input, model, model_loss
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from read_data import next_batch, reconstruct_image
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
slim = tf.contrib.slim
import os
import json

plt.interactive(False)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Model')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.997, help='batch norm decay argument for batch normalization.')
envarg.add_argument('--keep_prob', type=float, default=1.0, help='Dropout keep probability.')
envarg.add_argument("--is_training", type=int, default=True, help="Is training flag for batch normalization")
envarg.add_argument("--theta", type=float, default=1.0, help="Compression factor for the DenseNetwork 0 < θ ≤1.")
envarg.add_argument("--growth_rate", type=int, default=32, help="Growth rate for the DenseNetwork, the paper refars to it as the k parameter.")
envarg.add_argument("--number_of_classes", type=int, default=3, help="Number of classes to be predicted.")
envarg.add_argument("--aspp", type=bool, default=True, help="Use Atrous spatial pyrimid pooling.")
envarg.add_argument("--image_summary", type=bool, default=True, help="Activate tensorboard image_summary.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.1, help="starting learning rate.")
envarg.add_argument('--ending_learning_rate', type=float, default=0.00001, help="starting learning rate.")
envarg.add_argument("--channel_wise_inhibited_softmax", type=bool, default=True, help="Apply channel wise inhibited softmax.")
envarg.add_argument('--normalizer', choices=['standard', 'mean_subtraction'], default='mean_subtraction', help='Normalization option.')
envarg.add_argument('--upsampling_mode', choices=['resize', 'bilinear_transpose_conv'], default='resize', help='Upsampling algorithm.')

dataarg = parser.add_argument_group('Read data')
dataarg.add_argument("--crop_size", type=float, default=75, help="Crop size for batch training.")

trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=64, help="Batch size for network training.")
trainarg.add_argument("--total_epochs", type=int, default=550, help="Epoch total number for network training.")

args = parser.parse_args()

with open(log_folder + "/train" + 'data.json', 'w') as fp:
    json.dump(args, fp, sort_keys=True, indent=4)
exit()

log_folder = '/home/thalles_silva/log_folder'

# define the images and annotations path
base_dataset_dir = "/home/thalles_silva/DataPublic/Road_and_Buildings_detection_dataset/mass_merged"
train_dataset_base_dir = os.path.join(base_dataset_dir, "train")
images_folder_name = "sat/"
annotations_folder_name = "map/"
train_images_dir = os.path.join(train_dataset_base_dir, images_folder_name)
train_annotations_dir = os.path.join(train_dataset_base_dir, annotations_folder_name)

# read the train.txt file. This file contains the training images' names
file = open(os.path.join(train_dataset_base_dir, "train_all.txt"), 'r')
images_filename_list = [line for line in file]
number_of_train_examples = len(images_filename_list)
print("number_of_train_examples:", number_of_train_examples)

total_step = args.total_epochs*number_of_train_examples
print("Total steps:", total_step)

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

# define the images and annotations path
val_dataset_base_dir = os.path.join(base_dataset_dir, "valid")
val_images_dir = os.path.join(val_dataset_base_dir, images_folder_name)
val_annotations_dir = os.path.join(val_dataset_base_dir, annotations_folder_name)

# read the train.txt file. This file contains the training images' names
file = open(os.path.join(val_dataset_base_dir, "val.txt"), 'r')
val_images_filename_list = [line for line in file]

# Define the input shapes
input_image_shape = [args.crop_size, args.crop_size, 3]
label_image_shape = [args.crop_size, args.crop_size]

# get the model placeholders
batch_images_placeholder, batch_labels_placeholder, is_training_placeholder, keep_prob = model_input(input_image_shape, label_image_shape)

logits = model(batch_images_placeholder, args)

# get the error and predictions from the network
cross_entropy, pred, probabilities = model_loss(logits, batch_labels_placeholder, class_labels)

# Example: decay from 0.01 to 0.0001 in 10000 steps using sqrt (i.e. power=1. linearly):
global_step = tf.Variable(0, trainable=False)
decay_steps = total_step
learning_rate = tf.train.polynomial_decay(args.starting_learning_rate, global_step,
                                          decay_steps, args.ending_learning_rate,
                                          power=1)

with tf.variable_scope("adam_vars"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    gradients = optimizer.compute_gradients(loss=cross_entropy)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=global_step)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                   labels=batch_labels_placeholder,
                                                   num_classes=args.number_of_classes)


# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
log_folder = os.path.join(log_folder, process_str_id)
# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    print("Tensoboard folder:", log_folder)
    os.makedirs(log_folder)

saver = tf.train.Saver()

with tf.Session() as sess:
    # Create the summary writer -- to write all the logs
    # into a specified file. This file can be later read
    # by tensorboard.
    train_writer = tf.summary.FileWriter(log_folder + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_folder + '/test')

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.total_epochs):

        for batch_image, batch_annotations, _, _ in next_batch(train_images_dir, train_annotations_dir, images_filename_list,
                                                         batch_size=args.batch_size, crop_size=args.crop_size, normalizer=args.normalizer):

            _, global_step_np, train_loss, pred_np, probabilities_np, summary_string, lr_np = sess.run([train_step, global_step, cross_entropy, pred,
                                                                                 probabilities, merged_summary_op, learning_rate],
                                                        feed_dict={is_training_placeholder: args.is_training,
                                                                  batch_images_placeholder:batch_image,
                                                                  batch_labels_placeholder:batch_annotations,
                                                                  keep_prob: args.keep_prob})
            train_writer.add_summary(summary_string, global_step_np)

        if epoch % 10 == 0:
            total_miou_np = []
            total_val_loss = []

            for batch_images_val, batch_annotations_val, _, original_label in next_batch(val_images_dir, val_annotations_dir, val_images_filename_list,
                                                                      crop_size=args.crop_size, random_cropping=False, normalizer=args.normalizer):

                val_loss, pred_np, probabilities_np, summary_string, _ = sess.run([cross_entropy, pred, probabilities, merged_summary_op, update_op],
                                                                    feed_dict={is_training_placeholder: False,
                                                                              batch_images_placeholder:batch_images_val,
                                                                              batch_labels_placeholder:batch_annotations_val,
                                                                              keep_prob: 1.0})

                miou_np = sess.run(miou)
                total_val_loss.append(val_loss)
                total_miou_np.append(miou_np)

            # cmap = plt.get_cmap('bwr')
            # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            # f.set_figheight(10)
            # f.set_figwidth(28)
            #
            # ax1.imshow(original_label)
            # ax1.set_title('Ground Truth')
            # probability_graph = ax2.imshow(np.squeeze(reconstructed_image))
            # ax2.set_title('Predicted')
            # plt.show()

            print("Epoch:", epoch, "\tGlobal step:", global_step_np, "\tTraining loss:", train_loss, "\tValidation Loss:",
                  np.mean(total_val_loss), "\tmIOU:", np.mean(total_miou_np), "\tLearning rate:", lr_np)

            test_writer.add_summary(summary_string, global_step_np)

        # at the end epoch shuffle the dataset list files
        np.random.shuffle(images_filename_list)
        np.random.shuffle(val_images_filename_list)

    # Save the variables to disk.
    save_path = saver.save(sess, log_folder + "/train" + "/model.ckpt")

    with open(log_folder + "/train/" + 'data.json', 'w') as fp:
        json.dump(args.__dict__, fp, sort_keys=True, indent=4)

    train_writer.close()
