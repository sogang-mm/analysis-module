import os
# from Modules.dummy.example import test
import xml.etree.ElementTree as ET

class Dummy:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        model_path = os.path.join(self.path, "model.txt")
        self.model = open(model_path, "r")

    def inference_by_path(self, image_path):
        result = []
        # TODO
        #   - Inference using image path
        import time
        result = self.read_content(os.path.join(self.path, "./sample.xml"))
        # result = [[(0, 0, 0, 0), {'TEST': 0.95, 'DEBUG': 0.05}], [(100, 100, 100, 100), {'TEST': 0.95, 'DEBUG': 0.05}]]
        self.result = result

        return self.result


    def read_content(self, xml_file):

        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        list_with_all_boxes = []
        size = root.find('size')

        width = size.find('width').text
        height = size.find('height').text
        depth = size.find('depth').text

        patch = root.find('patch')
        patch_size = patch.find('size')
        patch_width = patch_size.find('width').text
        patch_height = patch_size.find('height').text

        img_size = {'width': width, 'height': height, 'depth': depth}
        filename = None

        for boxes in root.iter('object'):

            filename = root.find('filename').text

            ymin, xmin, ymax, xmax = None, None, None, None
            for box in boxes.findall('bndbox'):
                ymin = int(box.find('ymin').text)
                xmin = int(box.find('xmin').text)

            list_with_single_boxes = [(xmin, ymin, patch_width, patch_height), {'crack':100, 'none':0}]
            list_with_all_boxes.append(list_with_single_boxes)

        return list_with_all_boxes

dummy = Dummy()
print(dummy.inference_by_path('sample.jpg'))