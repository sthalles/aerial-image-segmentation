import importlib
import os
from fcn.fcn_8s import FCN_8s

model_name = "19209"

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from read_data import next_batch, reconstruct_image, central_crop
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
slim = tf.contrib.slim
import os
import itertools
import json
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, f1_score
plt.interactive(False)

log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)
args.crop_size = 1472 # FCN needs the input to be multiple of 32

#0:   background
#1:   roads
#2:   buildings
#255: undefined/don't care
number_of_classes = 3
class_labels = [v for v in range((number_of_classes+1))]
class_labels[-1] = 255

# Define the input shapes
input_image_shape = [args.crop_size, args.crop_size, 3]
label_image_shape = [args.crop_size, args.crop_size]

def model_input(input_image_shape, label_image_shape):
    # Create and return the placeholders for receiving data
    batch_images_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    batch_labels_placeholder = tf.placeholder(tf.uint8, [None, None, None])
    is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")
    return batch_images_placeholder, batch_labels_placeholder, is_training_placeholder

# get the model placeholders
batch_images_placeholder, batch_labels_placeholder, is_training_placeholder = model_input(input_image_shape, label_image_shape)

logits, _ = FCN_8s(batch_images_placeholder, args.number_of_classes, is_training_placeholder)

predictions = tf.argmax(logits, dimension=3)
probabilities = tf.nn.softmax(logits)

# accuracy, acc_update_op= tf.metrics.accuracy(batch_labels_placeholder, predictions)
#
# recall, recall_update_op = tf.metrics.recall(batch_labels_placeholder, predictions)
#
# # Define the accuracy metric: Mean Intersection Over Union
# miou, miou_update_op = slim.metrics.streaming_mean_iou(predictions=predictions,
#                                                    labels=batch_labels_placeholder,
#                                                    num_classes=number_of_classes)

# define the images and annotations path
base_dataset_dir = "/data/Road_and_Buildings_detection_dataset/mass_merged"
images_folder_name = "sat/"
annotations_folder_name = "map/"

# define the images and annotations path
val_dataset_base_dir = os.path.join(base_dataset_dir, "test")
val_images_dir = os.path.join(val_dataset_base_dir, images_folder_name)
val_annotations_dir = os.path.join(val_dataset_base_dir, annotations_folder_name)

# read the train.txt file. This file contains the train images' names
file = open(os.path.join(val_dataset_base_dir, "test.txt"), 'r')
val_images_filename_list = [line for line in file]

saver = tf.train.Saver()

test_folder = os.path.join(log_folder, model_name, "test")
train_folder = os.path.join(log_folder, model_name, "train")

with tf.Session() as sess:

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
    print("Model", model_name, "restored.")
    val_image_id = 0

    f = open(os.path.join(test_folder, "evaluation.txt"), 'w+')
    counter = 0
    for batch_images_val, batch_annotations_val, _, _ in next_batch(val_images_dir, val_annotations_dir, val_images_filename_list,
                                                              crop_size=args.crop_size, random_cropping=False,
                                                              perform_central_crop=True):


        pred_np, probabilities_np,logits_np = sess.run([predictions, probabilities,logits],
                                                            feed_dict={is_training_placeholder: False,
                                                                      batch_images_placeholder:batch_images_val,
                                                                      batch_labels_placeholder:batch_annotations_val})

        flatten_pred = pred_np.flatten()
        flatten_labels = batch_annotations_val.flatten()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(flatten_labels, flatten_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["background", "buildings", "roads"],
                      title='Confusion matrix, without normalization')

        cnf_matrix_file = val_images_filename_list[counter] + "_cnf_table"
        plt.savefig(os.path.join(test_folder, cnf_matrix_file), bbox_inches='tight')

        recall = recall_score(flatten_labels, flatten_pred, average=None)
        accuracy = accuracy_score(flatten_labels, flatten_pred)

        if pred_np.shape[0] > 1:
            reconstructed_image = reconstruct_image(pred_np, args.crop_size)
        else:
            reconstructed_image = pred_np

        #print(np.max(reconstructed_image))
        imsave(test_folder + "/" + val_images_filename_list[counter].strip() + "_pred_label.tiff", np.squeeze(reconstructed_image))
        imsave(test_folder  + "/" +  val_images_filename_list[counter].strip() + "_pred_heatmap.tiff", np.squeeze(logits_np))
        np.save(test_folder  + "/" +  val_images_filename_list[counter].strip() + "_prediction_np", pred_np)

        # cmap = plt.get_cmap('bwr')
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.squeeze(reconstructed_image), cmap=cmap)
        # ax1.set_title('Segmentation')
        # probability_graph = ax2.imshow((np.squeeze(batch_annotations_val)))
        # ax2.set_title('Ground-Truth Annotation')
        # fig.savefig(test_folder + "/pred_thruth" + str(val_image_id))

        val_image_id += 1

        metrics = "Accuracy: {0}\tClass Recall: {1}\n".format(accuracy,recall)
        print(metrics)
        counter+=1
        f.write(metrics)
    f.close()
