import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import numpy as np
import pathlib
import tensorflow as tf


#from models.research.object_detection.utils import label_map_util
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
#from models.research.object_detection.utils import colab_utils
from object_detection.builders import model_builder
import imutils

def building_model():
    tf.keras.backend.clear_session()
    
    num_classes = 1
    current_path = pathlib.Path().absolute()
    pipeline_config = str(current_path) + '/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    
    
    checkpoint_path = str(current_path) + '/checkpoints/ckpt-bottle_detection-1'

    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).

    # print(checkpoint_path)
    # exit()

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    print("building model")

    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )

    print("fake box predictor")
    
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    print("fake model")
    
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)

    # exit()

    ckpt.restore(checkpoint_path)
    return detection_model
