from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers.local import LocallyConnected2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import InceptionV3
from keras.layers import Input

from TrainSimulator import downsample_zeros, prepare_data, create_datasets, gen
from ImageProcessing import data_binning, transform_image, get_viewport, image_gradient, image_flip
from PIL import Image
from PIL import ImageOps
from numpy import genfromtxt
import numpy as np
import pickle
import cv2

########################################################################
# Simple Model
########################################################################
def simple_model(shape, n_classes):
    activation = 'relu'
    #Create the classifier
    classifier = Sequential()
    #Convolutional Layer
    classifier.add(Conv2D(16,1,1, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block0_conv0',
                                 input_shape=shape[1:]))

    classifier.add(Conv2D(32,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block1_conv1'))
    classifier.add(Activation(activation))
    classifier.add(AveragePooling2D(pool_size=(2,2), name='block1_pool'))

    classifier.add(Conv2D(64,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block2_conv1'))
    classifier.add(Activation(activation))
    classifier.add(AveragePooling2D(pool_size=(2,2), name='block2_pool'))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(128,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block3_conv1'))
    classifier.add(Activation(activation))
    classifier.add(AveragePooling2D(pool_size=(4,4), name='block3_pool'))
    classifier.add(Dropout(0.50))

    classifier.add(Conv2D(128,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block4_conv1'))

    classifier.add(Flatten(name="fc_flatten"))
    # Classifier Layer
    '''
    classifier.add(Dense(256, kernel_initializer='he_normal', name='fc_256'))
    classifier.add(Activation(activation))
    classifier.add(Dropout(0.25))

    classifier.add(Dense(128, kernel_initializer='glorot_normal', name='fc_256'))
    classifier.add(Activation(activation))
    classifier.add(Dropout(0.25))
    '''
    classifier.add(Dense(n_classes,
                         kernel_initializer='he_normal',
                         activation='softmax',
                         name='fc_output'))
    classifier.summary()
    return classifier


########################################################################
# Direct Drive Model: Take Advantage of SoftPlus activation
# characteristics (-1, +1 range) to generate steering angles directly.
# Single Neuron output
########################################################################
def direct_drive_model(shape, n_classes=1):
    activation = 'relu'
    #activation = 'tanh'
    #Create the classifier
    classifier = Sequential()
    #Convolutional Layer
    classifier.add(Conv2D(16,1,1, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block0_conv0',
                                 input_shape=shape[1:]))

    classifier.add(Conv2D(32,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block1_conv1'))
    classifier.add(Activation(activation))
    classifier.add(MaxPooling2D(pool_size=(2,2), name='block1_pool'))

    classifier.add(Conv2D(64,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block2_conv1'))
    classifier.add(Activation(activation))
    classifier.add(MaxPooling2D(pool_size=(2,2), name='block2_pool'))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(128,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block3_conv1'))
    classifier.add(Activation(activation))
    classifier.add(MaxPooling2D(pool_size=(4,4), name='block3_pool'))
    classifier.add(Dropout(0.50))

    classifier.add(Conv2D(128,3,3, kernel_initializer='glorot_normal',
                                 border_mode='same',
                                 name='block4_conv1'))

    classifier.add(Flatten(name="fc_flatten"))
    # Classifier Layer
    '''
    classifier.add(Dense(256, kernel_initializer='he_normal', name='fc_256'))
    classifier.add(Activation(activation))
    classifier.add(Dropout(0.25))

    classifier.add(Dense(128, kernel_initializer='glorot_normal', name='fc_128'))
    classifier.add(Activation(activation))
    classifier.add(Dropout(0.25))
    '''
    classifier.add(Dense(n_classes,
                         kernel_initializer='he_normal',
                         activation='softsign',
                         name='fc_output'))
    classifier.summary()
    return classifier


########################################################################
# AlexNet derivative
########################################################################
def ak_cifar10_model(shape, n_classes):
    activation = 'relu'
    classifier = Sequential()

    #Convolutional Layers
    classifier.add(Conv2D(64,(5,5),
                          padding='same', name='conv_1',
                          input_shape=shape[1:],
                          kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))
    classifier.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), name='conv_pool_1'))
    classifier.add(BatchNormalization(trainable=True))

    classifier.add(Conv2D(64,(5,5),
                          padding='same',
                          name='conv_2',
                          kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), name='conv_pool_2'))

    classifier.add(LocallyConnected2D(64,(3,3), name='lc_1', kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))

    classifier.add(LocallyConnected2D(64,(3,3), name='lc_2', kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))

    classifier.add(LocallyConnected2D(64,(2,2), name='lc_2', kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))

    classifier.add(Flatten(name="fc_flatten"))

    # Classifier Layer
    '''
    classifier.add(Dense(64, name='fc_1', kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))
    classifier.add(BatchNormalization())

    classifier.add(Dense(128, name='fc_2', kernel_initializer='glorot_normal'))
    classifier.add(Activation(activation))
    classifier.add(BatchNormalization())

    classifier.add(Dense(256, name='fc_3'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    '''
    classifier.add(Dense(n_classes, name='fc_out', kernel_initializer='glorot_normal'))
    classifier.add(Activation('softmax'))
    classifier.summary()
    #Save Model architecture to JSON
    model_json = classifier.to_json()
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    #for layer in classifier.layers:
    #    print("Layer {0} Trainable {1}".format(layer.name, layer.trainable))
    return classifier


########################################################################
# Final Combined model: Inception V3 + Classifier
########################################################################
def combined_model_inception(shape, n_classes):
    #Load Inception without FC layers and add AveragePooling
    #input_tensor = Input(shape=shape[1:])
    inceptionv3 = InceptionV3(input_shape=shape[1:], include_top=False)
    x = inceptionv3.output
    x = AveragePooling2D((2,2), name='inceptionv3_pool')(x)
    inceptionv3 = Model(inceptionv3.input, x)
    # i.e. freeze all Inception convolutional layers
    for inception_layer in inceptionv3.layers:
        inception_layer.trainable = False

    #Create the classifier
    classifier = Sequential()
    flatten_layer = Flatten(input_shape=inceptionv3.output_shape[1:])
    flatten_layer(x)
    classifier.add(flatten_layer)
    classifier.add(Dense(1024, init='normal', activation='relu', name='fc_1024'))
    #classifier.add(Dense(256, init='normal', activation='relu', name='fc_256'))
    classifier.add(Dense(n_classes, init='normal', activation='softmax', name='fc_output'))
    #Combined Model
    model = Model(inceptionv3.input, classifier.output)
    model.summary()
    return model


########################################################################
# Train the classifier
########################################################################
def train_classifier_original(model, x_train, y_train, x_val, y_val, n_classes, batch, epochs):
    # Compile and train the model here.
    train_gen = gen(x_train, y_train, batch, n_classes)
    val_gen = gen(x_val, y_val, batch, n_classes)
    #Expect the generator to create 4 additional views from each image
    samples    = 2*len(x_train)
    val_samples= 2*len(x_val)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    history = model.fit_generator(train_gen(),
                                  samples_per_epoch=samples,
                                  nb_epoch=epochs,
                                  verbose=1,
                                  validation_data=val_gen(),
                                  nb_val_samples=val_samples)
    model.save_weights('model_weights.h5')
    model.save('model.h5')


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

##############################################################################################
# Normalize batch of image with sample means
##############################################################################################
def preprocess_image_batch(x, means):
    if means is None:
        x = x/255 - 0.5
    else:
        x = x.astype("float32")
        #x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        r_mean, g_mean, b_mean = means[0], means[1], means[2]
        x[:, :, :, 0] -= r_mean
        x[:, :, :, 1] -= g_mean
        x[:, :, :, 2] -= b_mean
    return x

########################################################################
# Logits for training data
########################################################################
def sample_logits(data):
    # Generate mean, std across all training data for future sample normalization
    sample_means = np.zeros(3)
    sample_means[0] = np.mean(data[:, :, :, 0])
    sample_means[1] = np.mean(data[:, :, :, 1])
    sample_means[2] = np.mean(data[:, :, :, 2])
    return sample_means

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

            #x_batch = preprocess_image_batch(x_batch, means)
            # Categorical only necessary for multi-class model
            if n_classes > 1:
                y_batch = scale_labels(y_batch, n_classes)
                y_batch = to_categorical(y_batch, n_classes)
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size
            yield (x_batch, y_batch)
    return _f

# New improved trainer
def train_classifier(model, x_train, y_train, x_val, y_val, n_classes, batch=256, epochs=5):
    if n_classes > 1:
        loss_function = 'categorical_crossentropy'
        metrics = ['accuracy']
        monitor_metric = 'val_acc'
        filepath = "./training/weights-improvement-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5"
    else:
        # Direct drive uses softsign activation so train with mse loss
        loss_function = 'mse'
        metrics = ['mse']
        monitor_metric = 'val_loss'
        filepath = "./training/weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"

    model.compile(loss=loss_function,
                  optimizer=Adam(lr=0.001),
                  metrics=metrics)

    train_gen = gen(x_train, y_train, batch, n_classes)
    val_gen = gen(x_val, y_val, batch, n_classes)
    samples = 2*len(x_train)//batch - 10
    val_samples= 2 #2*len(x_val)//batch

    checkpoint = ModelCheckpoint(filepath,
                                 monitor=monitor_metric,
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.75,
                                  patience=2,
                                  min_lr=0.0001,
                                  mode='min',
                                  verbose=0)
    callbacks_list = [checkpoint,reduce_lr]
    history = model.fit_generator(train_gen(),
                                  steps_per_epoch=samples,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_data=val_gen(),
                                  validation_steps=val_samples,
                                  callbacks=callbacks_list)
    model.save_weights('model_weights.h5')
    model.save('model.h5')
    return



#################################################################
# MAIN Process
#################################################################
#######################################################
#
# MAIN APPLICATION
#
#######################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Trainer. Version 2.0')
    parser.add_argument('--mode',  help='simple|direct [direct]', type=str, default='direct')
    parser.add_argument('--batch', help='Batch Size [408]. Calculated at 8x n_classes for categorical', type=int, default=408)
    parser.add_argument('--epochs',help='Number of Training Epochs [80]', type=int, default=80)
    parser.add_argument('--test',  help='Run in test mode. This uses a subset of the training set [False]', action='store_true')
    parser.add_argument('--use_data_cache', help='Use prepared data [True]', action='store_false')
    parser.add_argument('--retrain',  help='Retrain an existing model [False]', action='store_true')
    args = parser.parse_args()

    mode = args.mode # simple|direct|ak|inception
    epochs = args.epochs
    default_shape = (None,112,112,3)
    use_prepared_data = args.use_data_cache
    test_mode = args.test
    retrain_mode = args.retrain
    
    if mode == 'direct':
        n_classes = 1
        batch = args.batch
    else:
        n_classes = 51
        batch = n_classes*8

    models = {
        'ak':ak_cifar10_model,
        'simple':simple_model,
        'inception': combined_model_inception,
        'direct': direct_drive_model
    }

    model_input = {
        'ak': (32,32),
        'simple': (40,40),
        'inception': (139,139),
        'direct': (40,40)
    }

    print("Loading data...")
    if use_prepared_data:
        print("Using Cache at training_data.p ...")
        data = pickle.load( open( "./training_data.p", "rb" ) )
        x_data, y_data = data['features'], data['labels']
    else:
        print("Preparing data at data/driving_log.csv ...")
        file_format = [('center','S64'),('left','S64'),('right','S64'),('steering', 'f8'),('throttle', 'f8')]
        metadata = genfromtxt('data/driving_log.csv', dtype=file_format, delimiter=',', skip_header=0, usecols=(0, 1, 2, 3, 4))

        print("Downsampling distribution to reduce zero bias")
        cleansed = downsample_zeros(metadata)
        print("Downsampled to {0}".format(len(cleansed)))

        print("Preparing data (crop, resize to model input size and pickle images. Optionally include side cameras...")
        x_data, y_data, xbw_data = prepare_data(cleansed, use_side_views=False, size=model_input[mode])
    print("Data load done.")
    print("_______________")

    print("Generating sample logits for image normalization (mean by color)")
    color_means = sample_logits(x_data)
    print("Means \tRed \tGreen \tBlue")
    print("{0:.2f}\t{1:.2f}\t{2:.2f}".format(color_means[0],color_means[1],color_means[2]))
    color_means = None
    x_data = preprocess_image_batch(x_data, color_means)
    print("_______________")

    #Split dataset
    print("Generating Full Training, Validation Data")
    x_train, y_train, x_val, y_val = create_datasets(x_data, y_data, n_classes=n_classes, test=False)
    print("Training Size:", len(x_train), "Validation Size:", len(x_val))
    print("_______________")

    ''' '''
    print("Building Classifier...")
    shape = (None,) + model_input[mode] + (3,)
    classifier = models[mode](shape, n_classes=n_classes)
    print("_______________")

    print("Training Classifier...")
    train_classifier(classifier,
                     x_train,
                     y_train,
                     x_val,
                     y_val,
                     n_classes=n_classes,
                     batch=batch,
                     epochs=epochs)
    print("_______________")
    print("Done!")
    print("_______________")
