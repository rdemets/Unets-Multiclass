import os

import glob
import datetime

import skimage.io
import numpy as np

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, ProgbarLogger

from .internals.image_functions import Image_Functions
from .internals.network_config import Network_Config
from .internals.dataset import Dataset

class CNN_Base(Dataset, Image_Functions):
    def __init__(self, model_dir = None, config_filepath = None, **kwargs):
        """Creates the base neural network class with basic functions

        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is stored
        config_filepath : `str`, optional
            [Default: None] Filepath to the config file
        **kwargs
            Parameters that are passed to :class:`network_config.Network_Config`

        Attributes
        ----------
        config : :class:`network_config.Network_Config`
            Network_config object containing the config and necessary functions
        """

        super().__init__()

        self.config = Network_Config(model_dir = model_dir, config_filepath = config_filepath, **kwargs)

        self.config.update_parameter(["general", "now"], datetime.datetime.now())

        if self.config.get_parameter("use_cpu") is True:
            self.initialize_cpu()
        else:
            self.initialize_gpu()

    #######################
    # Logging functions
    #######################
    def init_logs(self):
        """Initiates the parameters required for the log file
        """
        # Directory for training logs
        print(self.config.get_parameter("name"), self.config.get_parameter("now"))
        self.log_dir = os.path.join(self.config.get_parameter("model_dir"), "{}-{:%Y%m%dT%H%M}".format(self.config.get_parameter("name"), self.config.get_parameter("now")))

        if self.config.get_parameter("save_best_weights") is False:
            # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path = os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}_*epoch*.h5".format(self.config.get_parameter("name"), self.config.get_parameter("now")))
            self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        else:
            self.checkpoint_best = os.path.join(self.log_dir, "weights_best.h5")
            self.checkpoint_now = os.path.join(self.log_dir, "weights_now.h5")

    def write_logs(self):
        """Writes the log file
        """
        # Create log_dir if it does not exist
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)

        # save the parameters used in current run to logs dir
        self.config.write_config(os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}-config.yml".format(self.config.get_parameter("name"), self.config.get_parameter("now"))))

    #######################
    # Initialization functions
    #######################
    def summary(self):
        """Summary of the layers in the model
        """
        self.model.summary()

    def compile_model(self, optimizer, loss):
        """Compiles model
        Parameters
        ----------
        optimizer
            Gradient optimizer used in during the training of the network
        loss
            Loss function of the network
            
        metrics
            To try :
            
            Class tf.compat.v1.keras.metrics.MeanIoU
            Class tf.compat.v2.keras.metrics.MeanIoU
            Class tf.compat.v2.metrics.MeanIoU

        """
        if self.config.get_parameter("metrics") == ['IoU']:
            print("Metrics : IoU")
            from .internals.metrics import mean_iou
            self.model.compile(optimizer, loss = loss, metrics = [mean_iou]) 
            
            #self.model.compile(optimizer, loss = loss, metrics = [tf.keras.metrics.MeanIoU(num_classes=1+self.config.get_parameter("nb_classes"))]) 
        else:
            print("Metrics : {}".format(self.config.get_parameter("metrics")))
            self.model.compile(optimizer, loss = loss, metrics = self.config.get_parameter("metrics")) 

    def initialize_model(self):
        """Initializes the logs, builds the model, and chooses the correct initialization function
        """
        # write parameters to yaml file
        self.init_logs()
        if self.config.get_parameter("for_prediction") is False:
            self.write_logs()

        # build model
        self.model = self.build_model(self.config.get_parameter("input_size"))

        # save model to yaml file
        if self.config.get_parameter("for_prediction") is False:
            self.config.write_model(self.model, os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}-model.yml".format(self.config.get_parameter("name"), self.config.get_parameter("now"))))

        print("{} using single GPU or CPU..".format("Predicting" if self.config.get_parameter("for_prediction") else "Training"))
        self.initialize_model_normal()

    def initialize_cpu(self):
        """Sets the session to only use the CPU
        """
        config = tf.ConfigProto(
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
        session = tf.Session(config=config)
        K.set_session(session)

    def get_free_gpu(self):
        """Selects the gpu with the most free memory
        """
        import subprocess
        import os
        import sys
        from io import StringIO
        import numpy as np

        output = subprocess.Popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free', stdout=subprocess.PIPE, shell=True).communicate()[0]
        output = output.decode("ascii")
        # assumes that it is on the popiah server and the last gpu is not used
        memory_available = [int(x.split()[2]) for x in output.split("\n")[:-1]]
        print("Setting GPU to use to PID {}".format(np.argmax(memory_available)))
        return np.argmax(memory_available)

    def initialize_gpu(self):
        """Sets the seesion to use the gpu specified in config file
        """
        #if self.config.get_parameter("visible_gpu") == "None":
        #    gpu = self.get_free_gpu()
        #else:
        #    gpu = self.config.get_parameter("visible_gpu")

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) # needs to be a string
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0) # needs to be a string

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.tensorflow_backend.set_session(sess)

    def initialize_model_normal(self):
        """Initializes the optimizer and any specified callback functions
        """
        opt = self.optimizer_function()
        self.compile_model(optimizer = opt, loss = self.loss_function(self.config.get_parameter("loss")))

        if self.config.get_parameter("for_prediction") == False:
            self.callbacks = self.model_checkpoint_call(verbose = True)

            if self.config.get_parameter("use_tensorboard") is True:
                self.callbacks.append(self.tensorboard_call())

            if self.config.get_parameter("reduce_LR_on_plateau") is True:
                self.callbacks.append(ReduceLROnPlateau(monitor=self.config.get_parameter("reduce_LR_monitor"),
                                                        factor = self.config.get_parameter("reduce_LR_factor"),
                                                        patience = self.config.get_parameter("reduce_LR_patience"),
                                                        min_lr = self.config.get_parameter("reduce_LR_min_lr"),
                                                        verbose = True))

            if self.config.get_parameter("early_stopping") is True:
                self.callbacks.append(EarlyStopping(monitor=self.config.get_parameter("early_stopping_monitor"),
                                                    patience = self.config.get_parameter("early_stopping_patience"),
                                                    min_delta = self.config.get_parameter("early_stopping_min_delta"),
                                                    verbose = True))

    #######################
    # Optimizer/Loss functions
    #######################
    def optimizer_function(self, learning_rate = None):
        """Initialize optimizer function

        Parameters
        ----------
        learning_rate : `int`
            Learning rate of the descent algorithm

        Returns
        ----------
        optimizer
            Function to call the optimizer
        """
        if learning_rate is None:
            learning_rate = self.config.get_parameter("learning_rate")
        if self.config.get_parameter("optimizer_function") == 'sgd':
            return keras.optimizers.SGD(lr = learning_rate,
                                        decay = self.config.get_parameter("decay"),
                                        momentum = self.config.get_parameter("momentum"),
                                        nesterov = self.config.get_parameter("nesterov"))
        elif self.config.get_parameter("optimizer_function") == 'rmsprop':
            return keras.optimizers.RMSprop(lr = learning_rate,
                                            decay = self.config.get_parameter("decay"))
        elif self.config.get_parameter("optimizer_function") == 'adam':
            return keras.optimizers.Adam(lr = learning_rate,
                                         decay = self.config.get_parameter("decay"))

    def loss_function(self, loss):
        """Initialize loss function

        Parameters
        ----------
        loss : `str`
            Name of the loss function

        Returns
        ----------
        loss
            Function to call loss function
        """
        if loss == "binary_crossentropy":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_binary_crossentropy as loss
                print("Loss : edge-enhanced binary crossentropy")
            else:
                print("Loss : binary crossentropy")
            return loss
        elif loss == "categorical_crossentropy":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_categorical_crossentropy as loss
                print("Loss : Edge Enhanced categorical_crossentropy")
            else:
                print("ULoss : categorical_crossentropy")
            return loss
        elif loss == "jaccard_distance_loss":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_jaccard_distance_loss as jaccard_distance_loss
                print("Loss : edge-enhanced jaccard_distance_loss")
            else:
                print("Loss : jaccard distance loss")
                from .internals.losses import jaccard_distance_loss
            return jaccard_distance_loss
        elif loss == "dice_loss":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_dice_coef_loss as dice_coef_loss
                print("Loss : edge-enhanced Dice loss")
            else:
                print("Loss : Dice loss")
                from .internals.losses import dice_coef_loss 
            return dice_coef_loss
        elif loss == "bce_dice_loss":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_bce_dice_loss as bce_dice_loss
                print("Loss : Edge Enhanced 1 - Dice + BCE loss")
            else:
                print("Loss : 1 - Dice + BCE loss")
                from .internals.losses import bce_dice_loss
            return bce_dice_loss
        elif loss == "ssim_loss":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_DSSIM_loss as DSSIM_loss
                print("Loss : Edge Enhanced DSSIM loss")
            else:
                print("Loss : DSSIM loss")
                from .internals.losses import DSSIM_loss
            return DSSIM_loss
        elif loss == "bce_ssim_loss":
            if self.config.get_parameter("edge_enhance"):
                from .internals.losses import EE_bce_ssim_loss as bce_ssim_loss
                print("Loss : Edge Enhanced BCE + DSSIM loss")
            else:
                print("Loss : BCE + DSSIM loss")
                from .internals.losses import bce_ssim_loss
            return bce_ssim_loss

            
        elif loss == "mean_squared_error":
            return keras.losses.mean_squared_error
        elif loss == "mean_absolute_error":
            return keras.losses.mean_absolute_error
        
        elif loss == "lovasz_hinge":
            print("Loss : Lovasz-hinge loss")
            from .internals.losses import lovasz_loss
            return lovasz_loss   
        elif loss == "ssim_mae_loss":
            print("Loss : DSSIM + MAE loss")
            from .internals.losses import dssim_mae_loss
            return dssim_mae_loss
        else:
            print("Loss : {}".format(loss))
            return loss
        
        
    #######################
    # Callbacks
    #######################
    def tensorboard_call(self):
        """Initialize tensorboard call
        """
        return TensorBoard(log_dir=self.log_dir,
                           batch_size = self.config.get_parameter("batch_size_per_GPU"),
                           write_graph=self.config.get_parameter("write_graph"),
                           write_images=self.config.get_parameter("write_images"),
                           write_grads=self.config.get_parameter("write_grads"),
                           update_freq='epoch',
                           histogram_freq=self.config.get_parameter("histogram_freq"))

    def model_checkpoint_call(self, verbose = 0):
        """Initialize model checkpoint call
        """
        if self.config.get_parameter("save_best_weights") is False:
            return [ModelCheckpoint(self.checkpoint_path, save_weights_only=True, verbose=verbose)]
        else:
            return [ModelCheckpoint(self.checkpoint_best, save_best_only=True, save_weights_only=True, verbose=verbose),
                    ModelCheckpoint(self.checkpoint_now, save_weights_only=True, verbose=verbose)]

    #######################
    # Clear memory once training is done
    #######################
    def end_training(self):
        """Deletes model and releases gpu memory held by tensorflow
        """
        # del reference to model
        del self.model

        # clear memory
        tf.reset_default_graph()
        K.clear_session()

        # take hold of cuda device to shut it down
        from numba import cuda
        cuda.select_device(0)
        cuda.close()

    #######################
    # Train Model
    #######################
    def train_model(self, verbose = True):
        """Trains model

        Parameters
        ----------
        verbose : `int`, optional
            [Default: True] Verbose output
        """
        history = self.model.fit(self.aug_images, self.aug_ground_truth, validation_split = self.config.get_parameter("val_split"),
                                 batch_size = self.config.get_parameter("batch_size"), epochs = self.config.get_parameter("num_epochs"), shuffle = True,
                                 callbacks=self.callbacks, verbose=verbose)

        self.end_training()

    #######################
    # Predict using loaded model weights
    #######################
    # TODO: change to load model from yaml file
    def load_model(self, model_dir = None): # redo
        """Loads model from h5 file

        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Directory containing the model file
        """
        # TODO: rewrite to load model from yaml file
        if model_dir is None:
            model_dir = self.config.get_parameter("model_dir")

        if os.path.isdir(model_dir) is True:
            list_weights_files = glob.glob(os.path.join(model_dir,'*.h5'))
            list_weights_files.sort() # To ensure that [-1] gives the last file

            model_dir = os.path.join(model_dir,list_weights_files[-1])

        self.model.load_model(model_dir)
        print("Loaded model from: " + model_dir)

    def load_weights(self, weights_path = None, weights_index = -1):
        """Loads weights from h5 file

        Parameters
        ----------
        weights_path : `str`, optional
            [Default: None] Path containing the weights file or the directory to the weights file
        weights_index : `int`, optional
            [Default: -1]
        """
        if weights_path is None:
            weights_path = self.config.get_parameter("model_dir")

        if os.path.isdir(weights_path) is True:
            if self.config.get_parameter("save_best_weights") is True:
                weights_path = os.path.join(weights_path, "weights_best.h5")
            else:
                list_weights_files = glob.glob(os.path.join(weights_path,'*.h5'))
                list_weights_files.sort() # To ensure that [-1] gives the last file
                self.weights_path = list_weights_files[weights_index]
                weights_path = os.path.join(weights_path, self.weights_path)
        else:
            self.weights_path = weights_path

        self.model.load_weights(weights_path)
        print("Loaded weights from: " + weights_path)


    def predict_images(self, image_dir):
        """Perform prediction on images found in ``image_dir``

        Parameters
        ----------
        image_dir : `str`
            Directory containing the images to perform prediction on

        Returns
        ----------
        image : `array_like`
            Last image that prediction was perfromed on
        """
        
        # load image list
        from tqdm.notebook import tqdm
        image_list = self.list_images(image_dir)
        for image_path in tqdm(image_list):
        #for image_path in image_list:
            image = self.load_image(image_path = image_path)
            #print(image.shape)
            
            # percentile normalization
            if self.config.get_parameter("percentile_normalization"):
                image, _, _ = self.percentile_normalization(image, in_bound = self.config.get_parameter("percentile"))

            if self.config.get_parameter("tile_overlap_size") == [0,0]:
                padding = None
                if len(image.shape)==2:
                    image = np.expand_dims(image, axis = -1)       
                    
                # If length =3 : X Y C
                elif len(image.shape)==3:
                    if image.shape[0] != self.config.get_parameter("tile_size")[0]:
                        if image.shape[1] != self.config.get_parameter("tile_size")[1]:
                            image = np.transpose(image,(1,2,0)) 
                           
                    image = np.expand_dims(image, axis = 0)
                    if image.shape[1] < self.config.get_parameter("tile_size")[0] or image.shape[2] < self.config.get_parameter("tile_size")[1]:
                        image, padding = self.pad_image(image, image_size = self.config.get_parameter("tile_size"))
                
                # Else, length : N X Y Z / N X Y T
                elif len(image.shape)==4: 
                    if image.shape[1] != self.config.get_parameter("tile_size")[0]: # Means N X T Y
                        image = np.transpose(image,(0,1,3,2))
                    if image.shape[1] < self.config.get_parameter("tile_size")[0] or image.shape[2] < self.config.get_parameter("tile_size")[1]:
                        image, padding = self.pad_image(image, image_size = self.config.get_parameter("tile_size"))
                    #if image.shape[0] != 1:
                    #    image = np.transpose(image,(3,1,2,0))
                              
                
                # Single slice image vs Stack of images (no need of new axis)
                if len(image.shape)==3:
                    input_image = image[np.newaxis,:,:]
                    #output_image = self.model.predict(input_image, verbose=1)
                    output_image = self.model.predict(input_image)
                    
                elif len(image.shape)==4:
                    output_image = []
                    for i in tqdm(range(image.shape[0])):                        
                        input_image = image[i,:,:,:]
                        input_image = np.expand_dims(input_image, axis = 0)
                        if i == 0:
                            #output_image = self.model.predict(input_image, verbose=1)
                            output_image = self.model.predict(input_image)
                            
                        else:
                            #output_image = np.append(output_image,self.model.predict(input_image, verbose=1), axis = 0)
                            output_image = np.append(output_image,self.model.predict(input_image), axis = 0)
                            
                else:
                    output_image = image
                    for i in tqdm(range(image.shape[0])):
                        for j in range(image.shape[1]):
                            input_image = image[i,j,:,:,:]
                            input_image = np.expand_dims(input_image, axis = 0)
                            #output_image[i,j,:,:,:] = self.model.predict(input_image, verbose=1)
                            output_image[i,j,:,:,:] = self.model.predict(input_image)

                if padding is not None:
                    h, w = output_image.shape[1:3]
                    output_image = np.reshape(output_image, (h, w))
                    output_image = self.remove_pad_image(output_image, padding = padding)
            else:
                tile_image_list, num_rows, num_cols, padding = self.tile_image(image, self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"))

                pred_train_list = []
                for tile in tile_image_list:

                    # reshape image to correct dimensions for unet
                    h, w = tile.shape[:2]

                    tile = np.reshape(tile, (1, h, w, 1))

                    pred_train_list.extend(self.model.predict(tile, verbose=1))

                output_image = self.untile_image(pred_train_list, self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"),
                                                 num_rows, num_cols, padding = padding)

            self.save_image(output_image, image_path)
            #print(output_image.shape)

        return output_image

    def save_image(self, image, image_path, subfolder = 'Masks', suffix = '-preds'):
        """Saves image to image_path

        Final location of image is as follows:
          - image_path
              - subfolder
                 - model/weights file name

        Parameters
        ----------
        image : `array_like`
            Image to be saved
        image_path : `str`
            Location to save the image in
        subfolder : `str`
            [Default: 'Masks'] Subfolder in which the image is to be saved in
        suffix : `str`
            [Default: '-preds'] Suffix to append to the filename of the predicted image
        """
        image_dir = os.path.dirname(image_path)

        output_dir = os.path.join(image_dir, subfolder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.config.get_parameter("save_best_weights") is True:
            basename = os.path.basename(self.config.get_parameter("model_dir"))
        else:
            basename, _ = os.path.splitext(os.path.basename(self.weights_path))

        output_dir = os.path.join(output_dir, basename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename, _ = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(output_dir, "{}{}.tif".format(filename, suffix))

        if self.config.get_parameter("save_as_uint16") is True:
            image = skimage.util.img_as_uint(image)
        skimage.io.imsave(output_path, image)
