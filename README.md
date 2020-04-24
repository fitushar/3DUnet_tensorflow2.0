# 3DUnet_tensorflow2.0
This Repo is for implementation of 3D unet in Tensorflow 2.0v

## Files:
*    i) `unet_config.py -|--> All the Netword and Training configuration`
*   ii) `Unet3D          |--> Network architecture`
*  iii) `Train_Unet3D    |--> Training Script. it has tfrecord decoder, tfdataset reading pipeline and training loop,Losses and Matrics   function. Binary And Multi-class Dice Coefficent and Dice Loss`

## How to run
To run the model all is to need to configure the `unet_config.py` based on your requiremnet.
```ruby
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
TRAINING_SAVE_MODEL_PATH=''/Path/to/save/model/weight/Model.h5''
TRAINING_CSV='LungSEG_Model_March30_2020.csv'
####
TRAINING_TF_RECORDS='/Training/tfrecords/path/'
VALIDATION_TF_RECORDS='/Val/tfrecords/path/'
```

## Dice Loss
```ruby
def dice_coe(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))
```
