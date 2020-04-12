import numpy as np
import cv2
from mss import mss
from PIL import Image
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import pandas as pd
from keras_retinanet.models import load_model, convert_model
from test_model import show_detected_objects

def run_tester(model, labels_to_names):
	monitor = {
		'top': 0, 
		'left': 0, 
		'width': 1920 / 2, 
		'height': 1200 / 2
	}

	sct = mss()

	while True:
	    img = sct.grab(monitor)
	    # convert from sct image object to numpy
	    img = np.array(img)
	    # convert from BGRA format (sct default) to BGR (model input format)
	    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

	    img_w_boxes = show_detected_objects(img, model, labels_to_names)

	    cv2.imshow('test', img_w_boxes)

	    if cv2.waitKey(25) & 0xFF == ord('q'):
	        cv2.destroyAllWindows()
	        break

if __name__ == '__main__':
    CLASSES_FILE = 'class_names.csv'
    MODEL_FILE = 'models/resnet50_csv_20.h5'
    TEST_IMG_FILE = 'test_img.png'

    labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()

    model = load_model(MODEL_FILE, backbone_name='resnet50')
    # convert to inference model
    model = convert_model(model)

    run_tester(model, labels_to_names)
