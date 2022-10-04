# Has the content of .ipynb file without extra console outputs and image plots.
# Saves intermediate images as well as final result.

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import argparse
# Argparse block
parser = argparse.ArgumentParser(description='NN models description')
parser.add_argument('c_img', type=str, help='Content image input.')
parser.add_argument('s_img', type=str, help='Style image input.')
parser.add_argument('g_img', type=str, help='Generated image name.')
parser.add_argument('--num_iterations', action='store', type=int, default=200,
                    help='Number of iterations for the training')
parser.add_argument('--learning_rate', action='store', type=float, default=2, help='Learning rate for the training.')
parser.add_argument('--save_amount', action='store', type=int, default=20,
                    help='At each [save_amount] iteration an intermediate image would be saved.')
args = parser.parse_args()
from PIL import Image



import nst_utils
import importlib
importlib.reload(nst_utils)
from nst_utils import *
import numpy as np
import imageio.v2 as imageio
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Model loading
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

def compute_content_cost(a_C, a_G):
    C_shape, G_shape = list(tf.convert_to_tensor(a_C).shape.as_list()), list(tf.convert_to_tensor(a_G).shape.as_list())
    a_C = tf.reshape(a_C, [C_shape[0], C_shape[1]*C_shape[2], C_shape[3]])
    a_G = tf.reshape(a_G, [G_shape[0], G_shape[1]*G_shape[2], G_shape[3]])
    J_content = 1 / (4 * C_shape[1] * C_shape[2] * C_shape[3]) * tf.reduce_sum(tf.math.squared_difference(a_C, a_G))
    return J_content

def gram_matrix(A):
    return tf.linalg.matmul(A, A, transpose_b=True)

def compute_layer_style_cost(a_S, a_G):
    S_shape, G_shape = tf.convert_to_tensor(a_S).shape.as_list(), tf.convert_to_tensor(a_G).shape.as_list()
    a_S = tf.reshape(tf.convert_to_tensor(a_S), [S_shape[3], S_shape[1]*S_shape[2]])
    a_G = tf.reshape(tf.convert_to_tensor(a_G), [G_shape[3], G_shape[1]*G_shape[2]])
    S_gram = gram_matrix(a_S)
    G_gram = gram_matrix(a_G)
    J_style_layer = 1 / (4 * S_shape[1]**2 * S_shape[2]**2 * S_shape[3]**2) * tf.reduce_sum(tf.math.squared_difference(S_gram, G_gram))
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    return alpha * J_content + beta * J_style

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = Image.open("images/" + args.c_img)
content_image = np.asarray(content_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS)) # due to different sizes
content_image = content_image[:,:,:3] #For some png images. Can be removed!
content_image = reshape_and_normalize_image(content_image)

style_image = Image.open("images/" + args.s_img)
style_image = np.asarray(style_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS)) # due to different sizes
style_image = style_image[:,:,:3] #For some png images. Can be removed!
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations=args.num_iterations):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i % 5 == 0:
            print(str(i / num_iterations * 100) + "%...")
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            if i % args.save_amount == 0:
                save_image("output/" + args.g_img + str(i) + "_.png", generated_image)

    save_image('output/' + args.g_img, generated_image)

model_nn(sess, generated_image)