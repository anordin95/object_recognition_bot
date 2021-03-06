from keras_retinanet.models import load_model, convert_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

def predict(image, model):
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # boxes, scores, labels = model.predict_on_batch(
    # np.expand_dims(image, axis=0)
    # )
    boxes, scores, labels = model.predict(
        np.expand_dims(image, axis=0)
        )

    boxes /= scale

    return boxes, scores, labels

THRES_SCORE = 0.05
MAX_NUM_BOXES_PER_LABEL = {
    0 : 3 # at most 3 boxes for label 0: fishing_spot
}
def draw_detections(image, boxes, scores, labels, labels_to_names):
  num_boxes = 0
  for box, score, label in zip(boxes[0], scores[0], labels[0]):        
    # import pdb; pdb.set_trace()
    print(f"Score: {score}")

    max_num_boxes = MAX_NUM_BOXES_PER_LABEL.get(label, float('inf'))
    if score < THRES_SCORE or num_boxes >= max_num_boxes:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(image, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(image, b, caption)

    num_boxes += 1

def show_detected_objects(image, model, labels_to_names):
    boxes, scores, labels = predict(image, model)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # true_box = [
    # image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max
    # ]
    # draw_box(draw, true_box, color=(255, 255, 0))

    draw_detections(draw, boxes, scores, labels, labels_to_names)

    return draw
    
    # plt.axis('off')
    # plt.imshow(draw)
    # plt.savefig('fig.png')

if __name__ == '__main__':
    CLASSES_FILE = 'class_names.csv'
    MODEL_FILE = 'models/resnet50_csv_20.h5'
    TEST_IMG_FILE = 'test_img.png'

    labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()

    model = load_model(MODEL_FILE, backbone_name='resnet50')
    # convert to inference model
    model = convert_model(model)

    image = read_image_bgr(TEST_IMG_FILE)
    show_detected_objects(image, model)
