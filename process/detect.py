from process.obj_detection_model import building_model
import numpy as np
import tensorflow as tf




model = building_model()


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: a file path.

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    image = np.asarray(image)
    image = imutils.resize(image, width=500)
    image = Image.fromarray(image)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def ROI_extraction(detection_model, image):
    convertedToRGB = 0
    Realimage, image, got_roi = drawROI(image, detection_model)
    mask = colorMasking(image)

    if got_roi:
        if np.count_nonzero(mask) == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convertedToRGB = 1
            mask = colorMasking(image)
            if np.count_nonzero(mask) == 0:
                ROI = Realimage
            else:
                ROI = detectROI(mask, Realimage)
                if convertedToRGB == 0:
                    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)

        else:
            ROI = detectROI(mask, Realimage)
            if convertedToRGB == 0:
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
    else:
        ROI = Realimage

    return ROI

def detect(image):
    image_np = np.expand_dims(image, axis=0)
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)

    # print("prediction dict", prediction_dict)
    detection = model.postprocess(prediction_dict, shapes)

    label_id_offset = 1
    ROI_class_id = 1
    image_np_with_annotations = image_np[0].copy()

    category_index = {ROI_class_id: {'id': ROI_class_id, 'name': 'bottle'}}

    boxes = detection['detection_boxes'][0].numpy()
    classes = detection['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset
    scores = detection['detection_scores'][0].numpy()

    # print("boxes: ", boxes)
    # print("classes: ", classes)
    # print("scores: ", scores)
    # [[4.29540873e-02 2.89197952e-01 9.93089080e-01 7.18903899e-01]
    # (275, 183, 3)
    # 12, 52, 272, 128

    return boxes, scores