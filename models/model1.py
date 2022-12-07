import os

from . import nst_utils
import importlib

importlib.reload(nst_utils)
from .nst_utils import *
import numpy as np
from PIL import Image
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


PRETRAINED_MODELS_PATH = 'models/pretrained'

class Model1:
    def __init__(self):
        self.name = 'vgg-19'
        self.estimated_time_min = 3
        self.model = None
        self.sess = None
        self.STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)
        ]

    def load(self):
        self.model = load_vgg_model(f'{PRETRAINED_MODELS_PATH}/imagenet-vgg-verydeep-19.mat')

    def compute_content_cost(self, a_C, a_G):
        _, h, w, c = a_G.get_shape().as_list()
        a_C = tf.reshape(a_C, [c, h * w])
        a_G = tf.reshape(a_G, [c, h * w])
        J_content = (tf.reduce_sum((a_C - a_G) ** 2)) / (4 * h * w * c)
        return J_content

    def gram_matrix(self, A):
        return tf.linalg.matmul(A, A, transpose_b=True)

    def compute_layer_style_cost(self, a_S, a_G):
        _, h, w, c = a_G.get_shape().as_list()
        a_S = tf.transpose(tf.reshape(a_S, [h * w, c]))
        a_G = tf.transpose(tf.reshape(a_G, [h * w, c]))
        G_S = self.gram_matrix(a_S)
        G_G = self.gram_matrix(a_G)
        J_style_layer = (tf.reduce_sum((G_S - G_G) ** 2)) / (4 * h * w * c * c)
        return J_style_layer

    def compute_style_cost(self, model, STYLE_LAYERS):
        J_style = 0
        for layer_name, coeff in STYLE_LAYERS:
            out = model[layer_name]
            a_S = self.sess.run(out)
            a_G = out
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)
            J_style += coeff * J_style_layer
        return J_style

    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        return alpha * J_content + beta * J_style

    def run(self, content_image, style_image, learning_rate=2, num_iterations=200, save_amount=20):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        content_image = np.asarray(content_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS))  # due to different sizes
        content_image = content_image[:, :, :3]  # For some png images. Can be removed!
        content_image = reshape_and_normalize_image(content_image)

        style_image = np.asarray(style_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS))  # due to different sizes
        style_image = style_image[:, :, :3]  # For some png images. Can be removed!
        style_image = reshape_and_normalize_image(style_image)

        self.model = load_vgg_model(f"{PRETRAINED_MODELS_PATH}/imagenet-vgg-verydeep-19.mat")
        generated_image = generate_noise_image(content_image)
        self.sess.run(self.model['input'].assign(content_image))
        out = self.model['conv4_2']
        a_C = self.sess.run(out)
        a_G = out
        J_content = self.compute_content_cost(a_C, a_G)
        self.sess.run(self.model['input'].assign(style_image))
        J_style = self.compute_style_cost(self.model, self.STYLE_LAYERS)
        J = self.total_cost(J_content, J_style)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(J)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.model['input'].assign(generated_image))

        for i in range(num_iterations):
            self.sess.run(train_step)
            generated_image = self.sess.run(self.model['input'])

            if i % 5 == 0:
                print(str(i / num_iterations * 100) + "%...")
                _, _, _ = self.sess.run([J, J_content, J_style])
                if i % save_amount == 0:
                    save_image("output/" + str(i) + ".png", generated_image)

        self.sess.close()
        return generated_image

    def run_and_save(self, content_image_path: str, style_image_path: str, result_image_path: str):
        if not os.path.exists('output'):
            os.makedirs('output')
        content_image = Image.open(content_image_path)
        style_image = Image.open(style_image_path)
        result = self.run(content_image, style_image)
        save_image(result_image_path, result)


if __name__ == "__main__":
    pass
