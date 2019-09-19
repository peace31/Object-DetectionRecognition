# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model

if __name__ == '__main__':

    filename="images/samples/07647.jpg"
    img_width, img_height = 224, 224
    model = load_model()
    # model.load_weights('models/resnet152_weights_tf.h5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)


    image = cv.imread(filename)
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)),
                                0.007843, (300, 300), 127.5)
    net = cv.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.3:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            if(idx!=7):
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            bgr_img=image[startY:endY,startX:endX,:]
            bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = np.expand_dims(rgb_img, 0)
            preds = model.predict(rgb_img)
            prob = np.max(preds)
            class_id = np.argmax(preds)
            class_ids=np.argsort(preds)
            text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
            print(text)
    # results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
    # cv.imwrite('images/{}_out.png'.format(i), bgr_img)

    # print(results)
    # with open('results.json', 'w') as file:
    #     json.dump(results, file, indent=4)

    K.clear_session()
