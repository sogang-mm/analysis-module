import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

class Places:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.model = os.path.join(code_path,'resnet50.pth.tar')
        self.result = None
        
        self.model = torch.load(self.model)
        
        files = os.path.join(code_path,'categories.txt')

        self.classes = list()
        with open(files) as classes_file:
            for line in classes_file:
                self.classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(self.classes)        

    def inference_by_path(self, image_path):
        result = []
        
        # TODO
        #   - Inference using image path

        centre_crop = trn.Compose([
	        trn.CenterCrop(224),
	        trn.ToTensor(),
	        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img = Image.open(image_path)
	img = img.convert("RGB")
	input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

	logit = self.model.forward(input_img)
	h_x = F.softmax(logit, 1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	place_lists = [0, 0, img.width, img.height]
	#print('RESULT ON ' + img_name)
	lists = [place_lists] 

	a = {}
	for i in range(0, 5):
		print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
		a[self.classes[idx[i]]] = probs[i]
	lists.append(a)
	result = [lists]

        self.result = result

        return self.result

