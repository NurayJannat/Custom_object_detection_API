

def object_lists(boxes, scores, img_shape):
    result = []
    score = []

    for i in range(boxes.shape[0]):
        if scores[i] >= 0.4:
            y1 = int(img_shape[0]*boxes[i][0])
            x1 = int(img_shape[1]*boxes[i][1])
            y2 = int(img_shape[0]*boxes[i][2])
            x2 = int(img_shape[1]*boxes[i][3])
            result.append(
                {
                    "name": "bottle_" + str(i+1),
                    'x1': x1, 
                    'y1': y1, 
                    'x2': x2, 
                    'y2': y2, 
                    # 'score': scores[i]
                }
                
            )
            score.append(scores[i])
    return result, score