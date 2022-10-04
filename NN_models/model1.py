import nst_utils
import importlib
importlib.reload(nst_utils)
from nst_utils import *
import numpy as np
import imageio.v2 as imageio
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model1:
    def __init__(self):
        self.model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat") # Model loading
        self.STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]

    def compute_content_cost(self, a_C, a_G):
        C_shape, G_shape = list(tf.convert_to_tensor(a_C).shape.as_list()), list(tf.convert_to_tensor(a_G).shape.as_list())
        a_C = tf.reshape(a_C, [C_shape[0], C_shape[1]*C_shape[2], C_shape[3]])
        a_G = tf.reshape(a_G, [G_shape[0], G_shape[1]*G_shape[2], G_shape[3]])
        J_content = 1 / (4 * C_shape[1] * C_shape[2] * C_shape[3]) * tf.reduce_sum(tf.math.squared_difference(a_C, a_G))
        return J_content

    def gram_matrix(self, A):
        return tf.linalg.matmul(A, A, transpose_b=True)

    def compute_layer_style_cost(self, a_S, a_G):
        S_shape, G_shape = tf.convert_to_tensor(a_S).shape.as_list(), tf.convert_to_tensor(a_G).shape.as_list()
        a_S = tf.reshape(tf.convert_to_tensor(a_S), [S_shape[3], S_shape[1]*S_shape[2]])
        a_G = tf.reshape(tf.convert_to_tensor(a_G), [G_shape[3], G_shape[1]*G_shape[2]])
        S_gram = self.gram_matrix(a_S)
        G_gram = self.gram_matrix(a_G)
        J_style_layer = 1 / (4 * S_shape[1]**2 * S_shape[2]**2 * S_shape[3]**2) * tf.reduce_sum(tf.math.squared_difference(S_gram, G_gram))
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

    def get_res(self, c_img, s_img, learning_rate=2, num_iterations=200, save_amount=20):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        content_image = c_img
        content_image = np.asarray(content_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS))  # due to different sizes
        content_image = content_image[:, :, :3]  # For some png images. Can be removed!
        content_image = reshape_and_normalize_image(content_image)

        style_image = s_img
        style_image = np.asarray(style_image.resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS))  # due to different sizes
        style_image = style_image[:, :, :3]  # For some png images. Can be removed!
        style_image = reshape_and_normalize_image(style_image)

        self.model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
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
                J1, J2, J3 = self.sess.run([J, J_content, J_style])
                if i % save_amount == 0:
                    save_image("output/" + str(i) + ".png", generated_image)

        return generated_image

if __name__ == "__main__":
    pass
