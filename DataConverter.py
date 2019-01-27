
import numpy as np
import os
from tensorflow import keras
from PIL import Image

dimen = 28

dir_path = 'images/'
sub_dir_list = os.listdir( dir_path )
images = list()
labels = list()
for i in range( len( sub_dir_list ) ):
    label = i
    image_names = os.listdir( dir_path + sub_dir_list[i] )
    for image_path in image_names:
        path = dir_path + sub_dir_list[i] + "/" + image_path
        image = Image.open( path ).convert( 'L' )
        resize_image = image.resize((dimen, dimen))
        array = list()
        for x in range(dimen):
            sub_array = list()
            for y in range(dimen):
                sub_array.append(resize_image.load()[x, y])
            array.append(sub_array)
        image_data = np.array(array)
        image = np.array(np.reshape(image_data, (dimen, dimen, 1))) / 255
        images.append(image)
        labels.append( label )

x = np.array( images )
y = np.array( keras.utils.to_categorical( np.array( labels) , num_classes=2 ) )

num = 3000
limit = 10000

test_features = x[ 0 : num ]
test_labels = y[ 0 : num ]
train_features = x[ num : limit ]
train_labels = y[ num : limit ]

np.save( 'data28/x.npy' , train_features )
np.save( 'data28/y.npy' , train_labels )
np.save( 'data28/test_x.npy' , test_features )
np.save( 'data28/test_y.npy' , test_labels )



