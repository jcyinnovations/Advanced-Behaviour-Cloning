# ### Behaviour Cloning on Driving Simulator
# 
# First attempt using VGG16 and a custom classifier produced lackluster results but proved the viability of using a pretrained CNN.
# 
# This attempt will leverage the work of the NVIDIA team; who themselves rewrote the work on DARPA's DAVE. It will also include the following refinements:
# 
# - Use of all three cameras to greatly improve the ability of the model to recognize when off track and apply corrective action
# - Quantize steering values to 21 classes each in increments of 0.01 in the range -1.00 to +1.00.
# - Shuffle images while training to avoid overfitting to one class (e.g 0 steering is over-represented in the data
# - Downsample 0.00 steering to avoid overfitting to that in the model by only adding 1 in every 25 samples in a sequence of 0.00 steering samples
# - Augment other steering classes by shifting slightly left or right and adjusting the steering values to match. This should avoid memorizing images and allow the model to generalize a bit more.
# - Use Tensor Flow for 1-hot encoding steering values on the fly
# - Do augmentation on-the-fly with fit_generator and load images on the fly
# - Once this approach is proven, capture training data for the second track and refine the model with this new data

# ### 1. Data Preprocessing
# 
# - Load the driving log 
# - downsample 0.00 steering
# - Create a new array including L and R cameras in sequence
# - shuffle array
import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from ImageProcessing import data_binning, transform_image, get_viewport, image_gradient, image_flip
from PIL import Image
from PIL import ImageOps
import pickle
import cv2

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, AveragePooling2D, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.models import Model, Sequential
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical

####################
# Global variables
####################


##############################################################################################
# How to read image files
##############################################################################################
def image_file(name):
    return "data/{0}".format(name.decode("utf-8").strip())

##############################################################################################
# Convert steering angles to range of non-negative integers: 0 to (n_classes-1)
# Steering angles are in range -1 to +1
##############################################################################################
def scale_labels(y_data, n_classes):
    scaled_data = None
    if n_classes%2 > 0:
        scale = (n_classes-1)/2
        scaled_data = np.round(y_data*scale + scale)
    else:
        scale = n_classes/2
        scaled_data = np.round(y_data*scale)
    return scaled_data

#######################################################
# Check that the data is valid and readable.
#######################################################
def sample_logits(data):
    # Generate mean, std across all training data
    # for future sample normalization
    sample_means = np.zeros(3)
    count = len(data)
    for i in range(count):
        try:
            #print(data[i])
            img = mpimg.imread(image_file(data[i]))
            sample_means[0] += np.mean(img[:, :, 0])
            sample_means[1] += np.mean(img[:, :, 1])
            sample_means[2] += np.mean(img[:, :, 2])
        except:
            print("Item {0} Corrupted".format(i))
            count -= 1
    sample_means = sample_means/count
    print("Logits done.")
    return sample_means

######################################################################
# Downsample 0.00 steering by find sequences of 0.00 steering and
# keeping the first 2, middle 2 and last 2 images of the sequence
######################################################################
def downsample_zeros(data):
    removals = []
    cache = []
    for r in range(len(data)):
        if data[r]['steering'] == 0.00:
            cache.append(r)
        else:
            size = len(cache)
            if size > 8:
                #del cache[-1:]
                #del cache[:1]
                #rem = round(len(cache)/2)
                #del cache[rem-1:rem+1]
                # Remove remainding entries from the data
                #print("Removing", cache)
                removals = removals + cache
                cache = []
    return np.delete(data, removals, axis=0)


##############################################################################################
# Reduce the over-sampled features
##############################################################################################
def downsample(x_data, y_data, n_classes, target_count=0):
    bins, x_data_binned, y_data_binned, bin_count = data_binning(x_data, y_data, n_classes)
    # Sample metrics
    bin_count_array = np.array(bin_count)
    top25 = np.argpartition(bin_count_array, -2)[-25:]
    top_counts = bin_count_array[top25]
    cutoff = np.min(top_counts)
    if target_count==0:
        target_count = int(np.round(np.mean(bin_count)))
    #Downsample to mean
    print("Removing samples over target count...")
    removal_list = None
    for i in range(n_classes):
        if bin_count[i] > target_count:
            removal_count = int(bin_count[i] - target_count)
            print("removing",removal_count, " from class ", i)
            #generate a list of random removals
            removal_idx = np.random.choice(bin_count[i], size=removal_count, replace=False)
            if removal_list is None:
                removal_list = bins[i][removal_idx.tolist()]
            else:
                removal_list = np.append(removal_list, bins[i][removal_idx.tolist()])
    #Now remove the targeted samples
    print("removals", len(removal_list))
    x_data = np.delete(x_data, removal_list, axis=0)
    y_data = np.delete(y_data, removal_list, axis=0)
    return x_data, y_data

#########################################################################
# Prepare the images for training. This involves cropping and shrinking
# from 320x160 -> 320x80 -> 80x20 for faster training. Aspect ratio 
# is maintained just in case it affects recognition.
# The 320x80 viewport takes roughly the mid third of the image which 
# crops out everything above the horizon and the hood of the car
#########################################################################
def prepare_data(data, use_side_views=False, size=(80,20)):
    x_data  = data['center']
    y_data  = data['steering']
    xl_data = data['left']
    xr_data = data['right']

    # Use the L and R cameras
    x_cache = []
    xbw_cache = []
    y_cache = []
    std_data = np.std(y_data)
    correction = std_data/2
    for i in range(len(y_data)):
        #c,l,r = y_data[i], min(y_data[i]+correction, 1.0), max(y_data[i]-correction,-1.0)
        c,l,r = y_data[i], correction, -1*correction
        #adjust_steering(n_classes, y_data[i])
        #Read images
        x = mpimg.imread(image_file(x_data[i]))
        
        #Crop and Resize
        x_cropped = get_viewport(x, size=size)           #Size is changed for VGG16, size=(112,112)
        
        #Morphology filter to try training on edges only (b/w images)
        x_bw = image_gradient(x_cropped, channels=1)
        
        if use_side_views:
            #Load side views
            xl = mpimg.imread(image_file(xl_data[i]))
            xr = mpimg.imread(image_file(xr_data[i]))
            #Crop
            xl_cropped = get_viewport(xl)
            xr_cropped = get_viewport(xr)
            # grab image gradient
            xl_bw = image_gradient(xl_cropped, channels=1)
            xr_bw = image_gradient(xr_cropped, channels=1)
            #add to cache
            x_cache = x_cache + [x_cropped, xl_cropped, xr_cropped]
            xbw_cache = xbw_cache + [x_bw, xl_bw, xr_bw]
            y_cache = y_cache + [c, l, r]
        else:
            #Without L/R cameras
            x_cache = x_cache + [x_cropped]
            xbw_cache = xbw_cache + [x_bw]
            y_cache = y_cache + [c]
        # Progress report
        if i%500 == 0:
            print(i)
        else:
            if i%10 == 0:
                print(".", end="")
                
    y_upsampled = np.array(y_cache)
    x_upsampled = np.array(x_cache)
    xbw_upsampled = np.array(xbw_cache)

    #Pickle the augmented data to save time on training rounds
    pickle_dict = {'features': x_upsampled, 'labels': y_upsampled}
    pickle.dump(pickle_dict, open("./training_data.p", "wb"))
    pickle_dict = {'features': xbw_upsampled, 'labels': y_upsampled}
    pickle.dump(pickle_dict, open("./training_data_bw.p", "wb"))
    return x_upsampled, y_upsampled, xbw_upsampled

##############################################################################################
# Shuffle data and create validation and training sets
##############################################################################################
def create_datasets(x_data, y_data, n_classes, test=False):
    # Create Training and validation sets from the data
    x_train, y_train = shuffle(x_data, y_data)
    #Test Mode uses 10% of the available data
    if test:
        lt = int(len(x_train)/10)
        x_train = x_train[:lt]
        y_train = y_train[:lt]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
    return x_train, y_train, x_val, y_val

##############################################################################################
# Adjust steering based on viewport. Randomized normal multiplier to avoid clustering.
# 1/2 of full steering to make small adjustments only
##############################################################################################
def adjust_steering(n_classes, c_steer):
    mid = (n_classes-1)/2
    correction = round(0.10 * mid)
    l = c_steer + correction
    r = c_steer - correction
    return l, r

##############################################################################################
# Normalize the image with sample means
##############################################################################################
def preprocess_image(x, r_mean, g_mean, b_mean):
    x = x.astype("float32")
    #x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= r_mean
    x[:, :, 1] -= g_mean
    x[:, :, 2] -= b_mean
    return x

##############################################################################################
# Normalize batch of image with sample means
##############################################################################################
def preprocess_image_batch(x, means):
    x = x.astype("float32")
    #x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    r_mean, g_mean, b_mean = means[0], means[1], means[2]
    x[:, :, :, 0] -= r_mean
    x[:, :, :, 1] -= g_mean
    x[:, :, :, 2] -= b_mean
    return x

########################################################################
# This generator loads images and from disk and generate variants
########################################################################
def gen(data, labels, batch, n_classes):
    #2/3 of images in the batch come from L/R viewports
    batch_size = int(batch / 2) 
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        while True:
            x_batch, y_batch = data[start:end], labels[start:end]
            #print(x_batch.shape, y_batch.shape)
            #Balance the dataset by flipping samples
            flip_image_cache = []
            flip_label_cache = []
            for img, label in zip(x_batch, y_batch):
                img = cv2.flip(img,1)
                flip_image_cache = flip_image_cache + [img]
                label = -1 * label
                flip_label_cache = flip_label_cache + [label]
            x_batch = np.append(x_batch, flip_image_cache, axis=0)
            y_batch = np.append(y_batch, flip_label_cache)
            #print(x_batch.shape, y_batch.shape)
            
            #Preprocess data for training
            x_batch = x_batch/255 - 0.5
            y_batch = scale_labels(y_batch, n_classes)
            y_batch = to_categorical(y_batch, n_classes)
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size
            #print(start, end)
            yield (x_batch, y_batch)
    return _f

########################################################################
# Train the classifier
########################################################################
def train_classifier(model, x_train, y_train, x_val, y_val, n_classes, batch, epochs):
    # Compile and train the model here.
    train_gen = gen(x_train, y_train, batch, n_classes)
    val_gen = gen(x_val, y_val, batch, n_classes)
    #Expect the generator to create 4 additional views from each image
    samples    = 2*len(x_train)
    val_samples= 2*len(x_val)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    history = model.fit_generator(train_gen(), samples_per_epoch=samples, nb_epoch=epochs, verbose=1, validation_data=val_gen(), nb_val_samples=val_samples)
    model.save_weights('model_weights.h5')
    model.save('model.h5')


########################################################################
# Final Combined model: VGG16 + Classifier
########################################################################
def combined_model(shape, n_classes):
    #Load VGG16 without FC layers and add AveragePooling
    input_tensor = Input(shape=shape[1:])
    vgg16 = VGG16(input_tensor=input_tensor, include_top=False)
    x = vgg16.output
    x = AveragePooling2D((2,2), name='vgg_pool')(x)
    vgg16 = Model(vgg16.input, x)
    # i.e. freeze all VGG16 convolutional layers
    for vgg_layer in vgg16.layers:
        vgg_layer.trainable = False
    
    #Create the classifier
    classifier = Sequential()
    flatten_layer = Flatten(input_shape=vgg16.output_shape[1:])
    flatten_layer(x)
    classifier.add(flatten_layer)
    classifier.add(Dense(512, init='normal', activation='relu', name='fc_512'))
    classifier.add(Dense(256, init='normal', activation='relu', name='fc_256'))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(n_classes, init='normal', activation='softmax', name='fc_output'))
    # Load pretrained weights
    # topper.load_weights('fc_model.h5')
    model = Model(vgg16.input, classifier.output)
    model.summary()
    #Save combined model weights
    model.save_weights('model.h5')
    #Save Model architecture to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    return model

########################################################################
# Final Combined model: VGG16 + Classifier
########################################################################
def simple_model(shape, n_classes):
    #Create the classifier
    classifier = Sequential()
    #Convolutional Layer
    classifier.add(Convolution2D(32,3,3, init='normal', 
                                 activation='relu', 
                                 border_mode='same', 
                                 name='block1_conv1', 
                                 input_shape=shape[1:]))
    classifier.add(AveragePooling2D(pool_size=(2,2), name='block1_pool'))
    classifier.add(Convolution2D(64,3,3, init='normal', 
                                 activation='relu', 
                                 border_mode='same', 
                                 name='block2_conv1'))
    classifier.add(AveragePooling2D(pool_size=(2,2), name='block2_pool'))
    classifier.add(Dropout(0.25))    
    classifier.add(Flatten(name="fc_flatten"))
    # Classifier Layer
    classifier.add(Dense(256, init='normal', activation='relu', name='fc_512'))
    classifier.add(Dense(128, init='normal', activation='relu', name='fc_256'))
    #classifier.add(Dropout(0.25))
    classifier.add(Dense(n_classes, init='normal', activation='softmax', name='fc_output'))
    # Load pretrained weights
    classifier.summary()
    #Save combined model weights
    classifier.save_weights('model.h5')
    #Save Model architecture to JSON
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    return classifier

#######################################################
#
# MAIN APPLICATION
#
#######################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Trainer')
    parser.add_argument('--mode',  help='Select VGG16 (deprecated) or Simple model: vgg|simple [simple]', type=str, default='simple')
    parser.add_argument('--batch', help='Batch Size [512]', type=int, default=512)
    parser.add_argument('--epochs',help='Number of Training Epochs [20]', type=int, default=20)
    parser.add_argument('--test',  help='Run in test mode. This uses a subset of the training set [False]', action='store_true')
    args = parser.parse_args()
    
    #parser.add_argument('mode', type=str, help='Train full model, just the classifier or just export combined model: full|classifier|combined')
    #args = parser.parse_args()
    
    n_classes = 51
    default_shape = (None,112,112,3)
    
    print("Version 2.00")
    print("Loading data...")
    file_format = [('center','S64'),('left','S64'),('right','S64'),('steering', 'f8'),('throttle', 'f8')]
    metadata = genfromtxt('data/driving_log.csv', dtype=file_format, delimiter=',', skip_header=0, usecols=(0, 1, 2, 3, 4))
    
    print("Downsampling distribution to reduce zero bias")
    cleansed = downsample_zeros(metadata)
    print("Downsampled to {0}".format(len(cleansed)))
    
    print("Preparing data (crop, resize to 80x20 and pickle images Optionally include side cameras...")
    x_data, y_data, xbw_data = prepare_data(cleansed, use_side_views=False)
    print("Preparation done.")
    
    #print("Generating sample logits for image normalization (mean by color)")
    #color_means = sample_logits(x_data)
    #print("Means \tRed \tGreen \tBlue")
    #print(color_means[0],"\t",color_means[1],"\t",color_means[2])

    #Split dataset
    if args.test:
        print("Generating Sample Training, Validation Data in Test Mode (10%)")
    else:
        print("Generating Full Training, Validation Data")
    x_train, y_train, x_val, y_val = create_datasets(x_data, y_data, n_classes=n_classes, test=args.test)
    print("Training Size:", len(x_train), "Validation Size:", len(x_val))

    print("Building Classifier...")
    if args.mode == "vgg":
        shape = default_shape
        classifier = combined_model(shape, n_classes=n_classes)
    else:
        shape = (None,20,80,3)
        classifier = simple_model(shape, n_classes=n_classes)
    print("Training Classifier...")
    train_classifier(classifier, x_train, y_train, x_val, y_val, n_classes=n_classes, batch=args.batch, epochs=args.epochs)
    print("Done!")
