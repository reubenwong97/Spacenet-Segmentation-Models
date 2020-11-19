import random
import tensorflow as tf
from matplotlib import pyplot as plt

# Set seed for repeatable results
# seed = 4
# tf.random.set_seed(seed)

#np arrays must be rebuilt
def rot90(image,mask):
    k = random.randint(1,3)
    new_img = tf.image.rot90(image,k)
    new_mask = tf.image.rot90(mask,k)
    return new_img,new_mask

#np arrays must be rebuilt
def flip_lr(image,mask):
    new_image = tf.image.flip_left_right(image)
    new_mask = tf.image.flip_left_right(mask)
    return new_image,new_mask

def flip_ud(image,mask):
    new_image = tf.image.flip_up_down(image)
    new_mask = tf.image.flip_up_down(mask)
    return new_image, new_mask

#blend 2 images together
def blend(img1,img2):
    alpha = 0.7 + random.random() * 0.4
    #uint8_cast = tf.cast(img1*alpha + (1-alpha)*img2, dtype=tf.uint8)
    return img1*alpha + (1-alpha)*img2


#grayscale image only
def grayscale(image):
    alpha = tf.reshape(tf.constant([0.25,0.25,0.25]), (1,1,3))
    return tf.math.reduce_sum(alpha*image,axis=2,keepdims=True)

def saturation(image):
    gs = grayscale(image)
    return blend(image,gs)

def brightness(image):
    gs = tf.zeros_like(image)
    return blend(image,gs)

def gauss_noise(image):
    gauss = tf.random.normal(image.shape,10.0,10.0**0.5)
    new_image = tf.identity(image)
    new_image = tf.add(new_image,gauss)
    return new_image

def gamma(image):
    gamma = 0.7+0.4*random.random()
    new_image = tf.identity(image)
    new_image = tf.clip_by_value(new_image,clip_value_min=0.0,clip_value_max=255)
    new_image = tf.math.pow(new_image,gamma)
    return new_image

#Prob variables needed: rot90_prob,flipud_prob,fliplr_prob,color_aug(for brightness,
#contrast,saturation), gauss_prob, gamma_prob
def data_augment(image,mask,rot90_prob=0.4,flipud_prob=0.3,fliplr_prob=0.5,color_aug_prob=0.3,gauss_aug_prob=0.5,gamma_prob=0.2):
    if random.random() > 0.3:
        mask = tf.expand_dims(mask, -1)
        if random.random() < rot90_prob:
            image,mask = rot90(image,mask)
        if random.random() < fliplr_prob:
            image,mask = flip_lr(image,mask)
        if random.random() < flipud_prob:
            image,mask = flip_ud(image,mask)
        if random.random() < color_aug_prob:
            image = saturation(image)
        if random.random() < color_aug_prob:
            image = brightness(image)
        if random.random() < gauss_aug_prob:
            image = gauss_noise(image)
        if random.random() < gamma_prob:
            image = gamma(image)
        mask = tf.squeeze(mask,-1)
    return image,mask

# training_data = datagen.get_dataset('../data_project/train/SN_6.tfrecords')
# image_batch, label_batch = next(iter(training_data))
# sample_image = image_batch[100]
# sample_mask = label_batch[100]
# new_image,new_mask = data_augment(sample_image,sample_mask)
# #new_image = gamma(sample_image)
# print("new image shape:" ,new_image.shape)
# print("new_mask shape:",new_mask.shape)
# plt.subplot(221),plt.imshow(tf.cast(sample_image,tf.uint8)),plt.title('Input')
# plt.subplot(222),plt.imshow(tf.cast(new_image,tf.uint8)),plt.title('Output')
# plt.subplot(223),plt.imshow(tf.cast(sample_mask,tf.uint8)),plt.title('Input')
# plt.subplot(224),plt.imshow(tf.cast(new_mask,tf.uint8)),plt.title('Output')
# plt.show()
