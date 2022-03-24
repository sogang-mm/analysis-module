import torch
import os
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

class Places:
    def __init__(self):
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.model = os.path.join(code_path, 'place47.pth.tar')
        self.result = None
        self.model = torch.load(self.model)
        self.model = torch.nn.DataParallel(self.model).cuda()

        files = os.path.join(code_path, 'categories.txt')
        self.classes = list()
        with open(files) as classes_file:
            for line in classes_file:
                self.classes.append(line.strip().split(' ')[0][0:])
        self.classes = tuple(self.classes)

    def inference_by_path(self, image_path):
        result = []
        centre_crop = trn.Compose([
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path)
        img = img.convert("RGB")
        place_lists = [0, 0, img.width, img.height]
        img = img.resize((256, 256), Image.ANTIALIAS)
        input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        lists = [place_lists]
        a = {}
        for i in range(0, 5):
            print (probs[i], self.classes[idx[i]])
            a[self.classes[idx[i]]] = probs[i]
        lists.append(a)
        result = [lists]
        self.result = result

        return self.result
