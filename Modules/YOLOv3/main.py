import os
import pyyolo
import cv2

class YOLOv3:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        cfg_path = os.path.join(self.path, "yolov3-nsfw.cfg")
        model_path = os.path.join(self.path, "yolov3-nsfw.weights")
        data_path = os.path.join(self.path, "nsfw.data")
        self.model = pyyolo.YOLO(
                cfg_path,
                model_path,
                data_path,
                detection_threshold = 0.5,
                hier_threshold = 0.5,
                nms_threshold = 0.45
        )


    def inference_by_path(self, image_path):
        # TODO
        #   - Inference using image path
        image = cv2.imread(image_path)
        results = self.model.detect(image, rgb=False)

        result = []
        for i, obj in enumerate(results):
            print(obj.name)
            xmin, ymin, xmax, ymax = obj.to_xyxy()
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            result.append([(x,y,w,h), {obj.name: obj.prob}])

        self.result = result

        return self.result