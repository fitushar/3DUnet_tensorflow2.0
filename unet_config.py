import tensorflow as tf
import math
###---Number-of-GPU
NUM_OF_GPU=4
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1","gpu:2","gpu:3"]
'''
if want to resume training from the weights Set
RESUME_TRAINING=1
'''
###----Resume-Training
RESUME_TRAINING=1
RESUME_TRAIING_MODEL='/Path/of/the/model/weight/Model.h5'
TRAINING_INITIAL_EPOCH=1381
NUMBER_OF_CLASSES=1
INPUT_PATCH_SIZE=(384,192,192, 1)
##Training Hyper-Parameter
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
#TRAIN_CLASSIFY_LOSS=tf.keras.losses.binary_crossentropy()
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
#TRAIN_CLASSIFY_METRICS=tf.keras.metrics.binary_accuracy()
BATCH_SIZE=4
TRAINING_STEP_PER_EPOCH=math.ceil((76)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((6)/BATCH_SIZE)
TRAING_EPOCH=1600
NUMBER_OF_PARALLEL_CALL=4
PARSHING=2*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='LUNGSegVal_loss'
TRAINING_SAVE_MODEL_PATH='/Path/to/save/model/weight/Model.h5'
TRAINING_CSV='LungSEG_Model_March30_2020.csv'


####
TRAINING_TF_RECORDS='/Training/tfrecords/path/'
VALIDATION_TF_RECORDS='/Val/tfrecords/path/'
