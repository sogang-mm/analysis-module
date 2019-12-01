# import torch libraries
import shutil

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# import the utility functions
from .model import HED
import glob
import sys
import getopt
from functools import reduce
import json
import time
from multiprocessing import Process
from io import BytesIO
from datetime import datetime
import requests
import base64

class Segmentation:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.arg_Model = os.path.join(self.path, "HED_MUHAN_POT_BEST_191129.pth")
        self.max_width = 3704
        self.max_height = 10000

        self.arg_Thres = 200

        self.result = {}

        for Opt, Arg in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
            if Opt == '--model' and Arg != '': self.arg_Model = Arg
            if Opt == '--thres' and Arg != '': self.arg_Thres = float(Arg)

        # end
        self.evaluation = True  # our data : False !

        # fix random seed
        self.rng = np.random.RandomState(37148)

        # create instance of HED model
        self.model = HED()
        self.model.cuda()

        # load the weights for the model
        self.model.load_state_dict(torch.load(self.arg_Model))

    def inference_by_path(self, response):
        nBatch = 4
        json_file = open(response)
        json_array = json.load(json_file)
        image_url = json_array['image']
        image_name = image_url.split("/")[-1].split('.')[0]
        image_name = image_name + str(datetime.now().timestamp()).replace('.', '')

        outputDir_parent = os.path.join(self.path, 'slice/{}'.format(image_name))
        arg_DataRoot = os.path.join(self.path, 'slice', image_name, "test_256")
        arg_OutputDir = os.path.join(self.path, 'slice/{}/output{}'.format(image_name, self.arg_Thres))
        if os.path.exists('{}'.format(outputDir_parent)) == False:
            os.mkdir('{}'.format(outputDir_parent))
        if os.path.exists('{}'.format(arg_DataRoot)) == False:
            os.mkdir('{}'.format(arg_DataRoot))
        if os.path.exists('{}'.format(arg_OutputDir)) == False:
            os.mkdir('{}'.format(arg_OutputDir))

        # Patch Extraction
        self.extractPatch(json_array, arg_DataRoot, 'testimg', 256)

        # make test list for infer
        self.writeAll(arg_DataRoot)

        # create data loaders from dataset
        testPath = os.path.join(arg_DataRoot, 'test.lst')

        # # create data loaders from dataset
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        targetTransform = transforms.Compose([
            transforms.ToTensor()
        ])
        try :
            testDataset = self.TestDataset(testPath, arg_DataRoot, transform, targetTransform)

            testDataloader = DataLoader(testDataset, batch_size=nBatch)

            for i, sample in enumerate(testDataloader):
                # get input sample image
                if self.evaluation:
                    inp, fname = sample
                else:
                    inp, fname = sample
                input_data = Variable(inp.cuda())

                iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
                # perform forward computation
                s1, s2, s3, s4, s5, s6 = self.model.forward(input_data)

                start_time = time.time()

                outs = []
                s6_gray = self.grayTrans(s6.data.cpu())
                s1_gray = self.grayTrans(s1.data.cpu())
                s2_gray = self.grayTrans(s2.data.cpu())
                s3_gray = self.grayTrans(s3.data.cpu())
                s4_gray = self.grayTrans(s4.data.cpu())
                # convert back to numpy arrays

                for idx in range(s6_gray.shape[0]):
                    out = []
                    out.append(s6_gray[idx])  # 0
                    out.append(s1_gray[idx])  # 1
                    out.append(s2_gray[idx])  # 2
                    out.append(s3_gray[idx])  # 3
                    out.append(s4_gray[idx])  # 4
                    out.append(fname[idx])  # 5
                    out.append(inp[idx].cpu())  # 6

                    outs.append(out)

                procs = []

                for out in outs:
                    p = Process(target=self.Print, args=(out, arg_OutputDir))
                    procs.append(p)
                    p.start()

                # Print(arg_Thres,outs[0])
                for proc in procs:
                    proc.join()

            img_paths = glob.glob(os.path.join(arg_OutputDir, '*.png'))
            img_paths = sorted(img_paths)

            back = Image.new('RGB', (self.max_width, self.max_height), color='black')
            for idx, img_path in enumerate(img_paths):
                fname = img_path.split('/')[-1]
                fname_lst = fname.split('_')
                x_start = int(fname_lst[len(fname_lst) - 2])
                y_start = int(fname_lst[len(fname_lst) - 1].split('.')[0])
                print(x_start, y_start)
                img = Image.open(img_path)
                back.paste(img, (x_start, y_start))
        except :
            back = Image.new('RGB', (self.max_width, self.max_height), color='black')

        buffered = BytesIO()
        back.save(buffered, format="JPEG")
        result = base64.b64encode(buffered.getvalue()).decode('utf-8')

        self.result = result

        shutil.rmtree(outputDir_parent)

        return result

    def process_image(self, image_path):
        # image = cv2.imread(image_path), cv2.IMREAD_UNCHANGED)
        image = open(os.path.join(self.path, "./sample.png"),'rb')
        result_path = os.path.join(self.path, "./sample.png")
        # cv2.imwrite(result_path, image)
        return result_path

    def writeAll(self, dataRoot):
        txtfile = open(os.path.join(dataRoot, 'test.lst'), 'w')
        img_paths = glob.glob(os.path.join(dataRoot, 'testimg/*.jpg'))

        img_paths = sorted(img_paths)
        for img_path in img_paths:
            saved_img = os.path.relpath(img_path, dataRoot)
            saved_gt = saved_img.replace('testimg', 'testgt').replace('jpg', 'png')
            txtfile.write('{} {}\n'.format(saved_img, saved_gt))

        txtfile.close()

    def grayTrans(self, img):
        img = img.numpy()[:,0] * 255.0
        img = (img).astype(np.uint8)
        return img

    def union_intersect(self, true, pred,threshold=100):
        # Predict matrix, GT matrix vectorize for Intersection 1d , Union 1d, setDiff 1d Calculation
        h,w = true.shape
        nflat=true.ravel().shape

        pred = pred.copy()
        true = true.copy()

        pred=pred.astype(int)
        true=true.astype(int)

        pred[pred<threshold]=0
        pred[pred>=threshold]=255
        true_ravel = true.ravel()
        pred_ravel = pred.ravel()

        # Find index 255. or 1. region
        true_ind = np.where(true_ravel == 1)
        pred_ind = np.where(pred_ravel == 255)

        # Intersection , Union , Diff Calculation
        TP_ind = np.intersect1d(true_ind, pred_ind)
        FN_ind = np.setdiff1d(true_ind, TP_ind)
        FP_ind = np.setdiff1d(pred_ind,TP_ind)
        union_ind = reduce(np.union1d,(TP_ind, FN_ind, FP_ind))

        # Intersection of Union(HED,GT)


        TP_count = TP_ind.shape[0]
        union_count=union_ind.shape[0]
        pred_count = pred_ind[0].shape[0]
        true_count = true_ind[0].shape[0]

        precision = 0
        iou = 0
        recall = 0
        f1 = 0
        if TP_count==0 or pred_count==0 or true_count==0 or union_count==0:
            pass

        else :
            iou= TP_count / union_count
            precision = TP_count / pred_count
            recall = TP_count / true_count

            f1 = 2 * (precision * recall) / (precision + recall)

        # Create dummy array
        union = np.zeros(nflat)
        TP = np.zeros(nflat)
        FN = np.zeros(nflat)
        FP = np.zeros(nflat)

        # Write Array
        union[union_ind]=255
        TP[TP_ind]=255
        FN[FN_ind]=255
        FP[FP_ind]=255

        # return 2d arrays and iou
        return np.reshape(union,true.shape), np.reshape(TP,true.shape),np.reshape(FP,true.shape),np.reshape(FN,true.shape),precision,recall,iou ,f1

    def Print(self, out,outputDir):

        fname = out[5]
        inp = out[6]
        images=out[:5]
        output = Image.fromarray(out[0])


        if 'testimg' in fname.split('/'):
            save_path= (fname.split('.', 1)[0] + '.png').replace('testimg',outputDir)
            output.save(save_path)

    def extractPatch(self, json_array, dataRoot,outputDir,patchsize):
        saveDir = os.path.join(dataRoot,outputDir)
        if os.path.exists(saveDir) == False:
            os.mkdir(saveDir)

        image_url = json_array['image']

        cracks = json_array['results'][0]['module_result']

        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        self.max_width = int(img.size[0])
        self.max_height = int(img.size[1])
        image_name = image_url.split("/")[-1].split('.')[0]


        count = 0
        for j in range(0, len(cracks)):
            crack=cracks[j]
            labels = sorted(crack['label'], key=lambda label_list: (label_list['score']), reverse=True)
            if labels[0]['description'] == "pothole" :
                x = int(cracks[j]['position']['x'])
                y = int(cracks[j]['position']['y'])
                print(x, y)
                cropped_img=img.crop((x,y,x+patchsize,y+patchsize))
                saved_path = os.path.join(saveDir,'{}_{}_{}.jpg'.format(image_name,x,y))
                cropped_img.save(saved_path)
                count+=1

        print("num of patches: ", len(cracks), " / ", "only pots: ", count)

        return image_name

    class TestDataset(Dataset):
        def __init__(self, fileNames, rootDir, transform=None, target_transform=None):
            self.rootDir = rootDir
            self.transform = transform
            self.targetTransform = target_transform
            self.frame = pd.read_csv(fileNames, header=None, dtype=str, delimiter=' ')

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            # input and target images
            fname = self.frame.iloc[idx, 0]
            inputName = os.path.join(self.rootDir, fname)
            inputImage = Image.open(inputName).convert('RGB')

            if self.transform is not None:
                inputImage = self.transform(inputImage)

            return inputImage, fname

# if __name__ == '__main__':
#     json_file = open('document.json', encoding='utf-8')
#     response = json.load(json_file)
#
#     seg = Segmentation()
#     print(len(seg.inference_by_path(response)[0][1]['result']))


    # arg_Model : which model to use
    # arg_DataRoot : path to the dataRoot
    # arg_thres : threshold of the image output from the model














