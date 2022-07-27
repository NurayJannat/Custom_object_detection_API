import cv2

def draw(image, result, score):
    class_name = "bottle"

    for key in range(len(result)):
        print(key)
        image = cv2.rectangle(image, (result[key]['x1'], result[key]['y1']), (result[key]['x2'], result[key]['y2']), (0, 255, 0), 2)
        # image = cv2.putText(image, class_name+str(key), (result[key][0], result[key][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 0.1)
        print("rectangel")
        image = cv2.putText(image,class_name+str(key+1)+str(score[key]),(result[key]['x1'], result[key]['y1']),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        # image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        print("text")

    return image

    
