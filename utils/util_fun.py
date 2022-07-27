import numpy as np
import base64
import cv2

def base64_to_image(base64_image):
    try:
        encoded_data = base64_image.split(',')[1]
        
        decoded_data = base64.b64decode(encoded_data)
        np_data = np.fromstring(decoded_data,np.uint8)
        #img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        img = cv2.imdecode(np_data,flags=cv2.IMREAD_COLOR)
        
        #im_arr = np.frombuffer(im, dtype=np.uint8)  # im_arr is one-dim Numpy array
        #img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        #print("base64")


        return img

    except Exception as error:
        print('error_base64_to_image', error)
        # return error
        raise ValueError(str(error) + ". " + "Failed to convert base64 to image. Please provide a valid base64 image")


def image_to_base64(image):
    try:
        retval, buffer = cv2.imencode('.jpg', image)
        encoded_string = base64.b64encode(buffer)
        # encoded_string = base64.b64encode(image)
        # encoded_string = encoded_string.decode('utf-8')
        # print(encoded_string[:20])
        return encoded_string
    
    except Exception as error:
        print('error_base64', error)
        # return error
        raise ValueError(str(error) + ". " + "Failed to convert image to base64. Please provide a valid base64 image")