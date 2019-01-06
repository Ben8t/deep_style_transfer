import numpy as np
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
import time
import os
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Specify paths for 1) content image 2) style image and 3) generated image
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

content_image_path = "examples/content_image_1.jpg"
style_image_path = "examples/style_image_1.jpg"
generated_image_ouput_path = "results/generated_image_1.jpg"

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Image processing
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

def image_processing(image_path, target_size):
    image = load_img(path=image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = K.variable(preprocess_input(np.expand_dims(image_array, axis=0)), dtype='float32')  # adequate image to the format the model requires
    return image_array

target_heigth = 512
target_width = 512
target_size = (target_heigth, target_width)

content_image_origin = Image.open(content_image_path)
content_image_origin_size = content_image_origin.size

content_image_array = image_processing(content_image_path, target_size)
style_image_array = image_processing(style_image_path, target_size)

generated_image_0 = np.random.randint(256, size=(target_width, target_heigth, 3)).astype('float64')
generated_image_0 = preprocess_input(np.expand_dims(generated_image_0, axis=0))
g_img_placeholder = K.placeholder(shape=(1, target_width, target_heigth, 3))


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Define loss and helper functions
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

def get_feature_reps(x, layer_names, model):
    feature_matrices = []
    for layer_name in layer_names:
        selected_layer = model.get_layer(layer_name)
        feature_raw = selected_layer.output
        feature_raw_shape = K.shape(feature_raw).eval(session=tf_session)
        N_l = feature_raw_shape[-1]
        M_l = feature_raw_shape[1] * feature_raw_shape[2]
        feature_matrix = K.reshape(feature_raw, (M_l, N_l))
        feature_matrix = K.transpose(feature_matrix)
        feature_matrices.append(feature_matrix)
    return feature_matrices

def get_content_loss(F, P):
    content_loss = 0.5 * K.sum(K.square(F - P))
    return content_loss

def get_gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    style_loss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_gram_matrix(G)
        A_gram = get_gram_matrix(A)
        style_loss += w * 0.25 * K.sum(K.square(G_gram - A_gram)) / (N_l**2 * M_l**2)
    return style_loss

def get_total_loss(g_img_placeholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(g_img_placeholder, layer_names=[content_layer_names], model=g_model)[0]
    Gs = get_feature_reps(g_img_placeholder, layer_names=style_layer_names, model=g_model)
    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

def calculate_loss(g_img_array):
    """
    Calculate total loss using K.function
    """
    if g_img_array.shape != (1, target_width, target_width, 3):
        g_img_array = g_img_array.reshape((1, target_width, target_heigth, 3))
    loss_function = K.function([g_model.input], [get_total_loss(g_model.input)])
    return loss_function([g_img_array])[0].astype('float64')

def get_grad(g_img_array):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if g_img_array.shape != (1, target_width, target_heigth, 3):
        g_img_array = g_img_array.reshape((1, target_width, target_heigth, 3))
    gradient_function = K.function([g_model.input], K.gradients(get_total_loss(g_model.input), [g_model.input]))
    grad = gradient_function([g_img_array])[0].flatten().astype('float64')
    return grad

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (target_width, target_heigth, 3):
        x = x.reshape((target_width, target_heigth, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def reprocess_array(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x

def save_original_size(x, image_ouput_path, target_size=content_image_origin_size):
    x_image = Image.fromarray(x)
    x_image = x_image.resize(target_size)
    x_image.save(image_ouput_path)
    return x_image

tf_session = K.get_session()
content_model = VGG16(include_top=False, weights='imagenet', input_tensor=content_image_array)
style_model = VGG16(include_top=False, weights='imagenet', input_tensor=style_image_array)
g_model = VGG16(include_top=False, weights='imagenet', input_tensor=g_img_placeholder)
content_layer_names = 'block4_conv2'
style_layer_names = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                #'block5_conv1'
                ]

P = get_feature_reps(x=content_image_array, layer_names=[content_layer_names], model=content_model)[0]
As = get_feature_reps(x=style_image_array, layer_names=style_layer_names, model=style_model)
ws = np.ones(len(style_layer_names))/float(len(style_layer_names))

iterations = 600
x_val = generated_image_0.flatten()
start = time.time()

# TODO: add in optimizer callback=save_image
xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                            maxiter=iterations, disp=True) 
x_out = postprocess_array(xopt)
x_image = save_original_size(x_out, generated_image_ouput_path)
print("Image saved")
end = time.time()
print("Time taken: {}'".format(end-start))

