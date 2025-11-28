import os
import cv2
import pandas as pd

class SSD3:

    folder = 'cv_models/ODM/SSD3'
    sub_classes = [1,2,3,4,6,7,8,9,10,11,13,14,15]

    def __init__(self, input_size = (320, 320)):
        model_file = f'{SSD3.folder}/model/frozen_inference_graph.pb'
        config = f'{SSD3.folder}/config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        classes_csv = f'{SSD3.folder}/model/classes.csv'

        self.model = cv2.dnn_DetectionModel(model_file, config)
        self.classes = pd.read_csv(classes_csv, index_col = 'id')
        self.sub_classes = self.classes.loc[SSD3.sub_classes]
        
        self.model.setInputSize(*input_size)
        self.model.setInputScale(1.0/127.5) ## 255/2 = 127.5
        self.model.setInputMean((127.5, 127.5, 127.5)) ## MobileNet => [-1, 1]
        self.model.setInputSwapRB(True)
    
    def count_classes(self, image, conf = 0.6, sub_classes = True):
        counts = []
        not_found = False
        if type(image) is str:
            if os.path.exists(image):
                img = cv2.imread(image)
                classes, confidences, boxes = self.model.detect(img, confThreshold = conf)
                if len(classes)>0:
                    counts = list(classes.flatten())
                else:
                    counts = []
            else:
                not_found = True

        elif type(image) is list:
            for im in image:
                if os.path.exists(im):
                    img = cv2.imread(im)
                    classes, confidences, boxes = self.model.detect(img, confThreshold = conf)
                    if len(classes)>0:
                        counts += list(classes.flatten())
                else:
                    not_found = True

        # Nothing was counted
        if not_found:
            if sub_classes:
                df = self.sub_classes.copy()
                df['counts'] = -1
            else:
                df = self.classes.copy()
                df['counts'] = -1

        elif len(counts) == 0:
            if sub_classes:
                df = self.sub_classes.copy()
                df['counts'] = 0
            else:
                df = self.classes.copy()
                df['counts'] = 0

        else:
            df = pd.DataFrame({'id': counts, 'counts': 1})
            df = df.groupby(by = 'id').count()

            if sub_classes:
                df = self.sub_classes.join(df, how = 'left').fillna(0)
            else:
                df = self.classes.join(df, how = 'left').fillna(0)
        return df






