import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
# from scipy.misc import imread

from Modules.DAFRCNN.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from Modules.DAFRCNN.lib.model.rpn.bbox_transform import clip_boxes
from Modules.DAFRCNN.lib.model.nms.nms_wrapper import nms
from Modules.DAFRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv
from Modules.DAFRCNN.lib.model.utils.net_utils import save_net, load_net, vis_detections
from Modules.DAFRCNN.lib.model.utils.blob import im_list_to_blob
from Modules.DAFRCNN.lib.model.faster_rcnn.vgg16 import vgg16
from Modules.DAFRCNN.lib.model.faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


class DAFRCNN:
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.vis = False
        self.cuda = True
        self.class_agnostic = False
        self.classes = np.asarray(['__background__',  # always index 0
                                   'bottle', 'chair', 'cup or mug', 'flower pot', 'person', 'sofa', 'table', 'tie'])
        self.thresh = 0.8

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        cfg_file = os.path.join(self.path, "cfgs/res101.yml")
        if cfg_file is not None:
            cfg_from_file(cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)

        # TODO
        #   - initialize and load model here
        model_path = os.path.join(self.path, 'faster_rcnn_1_7_15425.pth')
        checkpoint = torch.load(model_path)
        self.model = resnet(self.classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        self.model.create_architecture()
        self.model.load_state_dict(checkpoint['model'])
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.eval()

    def inference_by_path(self, image_path):
        result = []
        # TODO
        #   - Inference using image path
        # Load the demo image
        im_in = cv2.imread(image_path)
        # im_in = np.array(imread(image_path))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        # im = im_in[:, :, ::-1]
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        if self.cuda:
            im_data = Variable(im_data_pt.cuda())
            im_info = Variable(im_info_pt.cuda())
            gt_boxes = Variable(torch.zeros([1,1,5]).cuda())
            num_boxes = Variable(torch.zeros([1]).cuda())
        else:
            im_data = Variable(im_data_pt)
            im_info = Variable(im_info_pt)
            gt_boxes = Variable(torch.zeros([1,1,5]))
            num_boxes = Variable(torch.zeros([1]))

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.model(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    #assert False, box_deltas.size()
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
            pred_boxes = _.cuda() if self.cuda > 0 else _

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        if self.vis:
            im2show = np.copy(im_in)
        for j in xrange(1, len(self.classes)):
            inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                for d in range(cls_dets.size(0)):
                    res = [(cls_dets[d][0], cls_dets[d][1], cls_dets[d][2], cls_dets[d][3]),
                           {str(self.classes[j]): cls_dets[d][4]}]
                    result.append(res)
                if self.vis:
                    im2show = vis_detections(im2show, self.classes[j], cls_dets.cpu().numpy(), 0.5)
        if self.vis:
            import matplotlib.pyplot as plt
            plt.imshow(im2show)
            plt.show()

        self.result = result
        return self.result
