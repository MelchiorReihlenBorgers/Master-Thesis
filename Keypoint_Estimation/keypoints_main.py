import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from DataLoad import DataLoad

from Keypoint_Estimation.keras_DataGenerator import DataGenerator
from Keypoint_Estimation.gaussian_heatmap import gaussian

from tensorflow.keras import models, layers

# Load images
data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
n_total = len(data.extract_paths()[ 0 ])  # If you want to load all images.
images, depths, radians = data.load_data(N=2)

path = "/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/labels_keypoints.csv"

keypoint_data = pd.read_csv(path,
                            header = 0,
                            names = ["Label", "X", "Y", "Image_Name", "Xdim", "Ydim"])


keypoint_data = keypoint_data.pivot_table(index = "Image_Name", columns = "Label", values = ["X", "Y"]).bfill()

keypoint_data = np.array(keypoint_data)

example_kp = keypoint_data[0:1]

X_batch, [y_batch, _] = next(DataGenerator(imgs = images[0:2], kps = example_kp, batch_size=1).__iter__())


#Plot a single image overlayed with its heatmaps
plt.imshow(X_batch[0].reshape(3024,4032, 3), alpha=0.5)
plt.imshow(y_batch[0].sum(axis=2), alpha=0.5)

####################### COPY ##################################
def conv_block(x, nconvs, n_filters, block_name, wd=None):
    for i in range(nconvs):
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name=block_name + "_pool")(x)
    return x


# Represents one stage of the model
def stages(x, stage_num, num_keypoints=8):
    # Block 1
    x = conv_block(x, nconvs=2, n_filters=64, block_name="block1_stage{}".format(stage_num))

    # Block 2
    x = conv_block(x, nconvs=2, n_filters=128, block_name="block2_stage{}".format(stage_num))

    # Block 3
    pool3 = conv_block(x, nconvs=3, n_filters=256, block_name="block3_stage{}".format(stage_num))

    # Block 4
    pool4 = conv_block(pool3, nconvs=3, n_filters=512, block_name="block4_stage{}".format(stage_num))

    # Block 5
    x = conv_block(pool4, nconvs=3, n_filters=512, block_name="block5_stage{}".format(stage_num))

    # Convolution 6
    x = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu",
               name="conv6_stage{}".format(stage_num))(x)

    # Convolution 7
    x = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu",
               name="conv7_stage{}".format(stage_num))(x)

    # upsampling
    preds_pool3 = Conv2D(8, kernel_size=(1, 1), strides=1, padding="same",
                         name="preds_pool3_stage{}".format(stage_num))(x)
    preds_pool4 = Conv2D(8, kernel_size=(1, 1), strides=1, padding="same",
                         name="preds_pool4_stage{}".format(stage_num))(preds_pool3)
    up_pool4 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation='relu',
                               name="ConvT_pool4_stage{}".format(stage_num))(preds_pool4)
    up_conv7 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation='relu',
                               name="ConvT_conv7_stage{}".format(stage_num))(up_pool4)

    heatmaps = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu',
                               name="convT_fusion_stage{}".format(stage_num))(up_conv7)
    heatmaps = Conv2D(num_keypoints, kernel_size=(1, 1), strides=1, padding="same", activation="linear",
                      name="output_stage{}".format(stage_num))(heatmaps)
    return heatmaps


# Create a single stage FCN
def build_model(input_shape):
    outputs = [ ]

    img = Input(shape=input_shape, name="Input_stage")

    ### Stage 1 ###
    heatmaps1 = stages(img, 1)
    outputs.append(heatmaps1)

    model = Model(inputs=img, outputs=outputs, name="FCN_Final")
    return model


# Training the model using mean squared losss
def get_loss_func():
    def mse(x, y):
        return mean_squared_error(x, y)

    keys = [ 'output_stage1', 'output_stage2' ]
    losses = dict.fromkeys(keys, mse)
    return losses


model = build_model((3024, 4032, 3))
losses = get_loss_func()
model.compile(loss=losses, optimizer='adam')
model.summary()

################## END COPY #####################

"""
Idea:
    - Dependent variable is heatmap of size 72 x 3024 x 4032 x 1
    - Feature is image of size 72 x 3024 x 4032 x 1
"""


n_images, cols = keypoint_data.shape
cols = int(cols/2)

n_images = 1
cols = 2

heatmaps = []
for col in range(int(cols)):
    for kp in range(n_images):
        heatmap = gaussian(xL = keypoint_data[kp,col], yL = keypoint_data[kp,col+8], sigma=100, H = 3024, W = 4032)
        heatmaps.append(heatmap)
        print("{} heatmaps are created".format(len(heatmaps)))

# Reshape the heatmaps list properly to ndarray of shape n_images x height x width x n_keypoints
array_heatmaps = np.array(heatmaps).reshape((n_images, cols, 3024, 4032)).swapaxes(1,3).swapaxes(1,2)
print("Shape of the heatmaps: {}".format(np.shape(array_heatmaps)))

# Plot one example
for i in range(cols):
    plt.imshow(array_heatmaps[0,:,:,i], alpha = 0.5)

plt.title("Keypoints of the first image")
plt.show()

X = np.reshape(images[0], (1, 3024, 4032, 3))

################## My Tensorflow Model ###################
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(3024, 4032, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2DTranspose(32, kernel_size=1, strides = (1,1)))
model.add(layers.Conv2DTranspose(32, kernel_size=2, strides = 2))
model.add(layers.Conv2DTranspose(32, kernel_size=4, strides = 4))
model.add(layers.Conv2DTranspose(32, kernel_size=19, strides = (1,1)))
model.add(layers.Conv2D(2, (3,3), activation= "linear"))


model.summary()

model.compile(optimizer="adam", loss = tf.keras.losses.MeanSquaredError())

model.fit(x = X, y = array_heatmaps)
