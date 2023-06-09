from labels import COCOLabels
import numpy as np
import json
import sys
import os

from pathlib import Path
from typing import List

from triton_python_backend_utils import get_input_tensor_by_name, get_output_config_by_name, triton_string_to_numpy, get_input_config_by_name
from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest

sys.path.append(Path(__file__).parent.absolute().as_posix())

def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=float)
    return anchors.reshape(3, 3, 2)

def expit(x):
    return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    '''define anchor boxes'''
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        print(">>>>>>>conv_shape: ", conv_shape)
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(float)

        pred_xy = ((expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(float).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=float)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'Identity:0': 'Identity:0',
            'Identity_1:0': 'Identity_1:0',
            'Identity_2:0': 'Identity_2:0',
            'orig_img_hw': 'input_orig_img_hw'
        }
        self.output_names = {
            'bboxes': 'output_bboxes',
            'labels': 'output_labels'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')
        if len(model_config['input']) != 4:
            raise ValueError(
                f'Expected 3 inputs, got {len(model_config["input"])}')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')
        if len(model_config['output']) != 2:
            raise ValueError(
                f'Expected 2 outputs, got {len(model_config["output"])}')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found '
                                     f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}
            max_num_bboxes_in_single_img = 0

            for out1, out2, out3, orig_img_hw in zip(batch_in['Identity:0'], batch_in['Identity_1:0'], batch_in['Identity_2:0'], batch_in['orig_img_hw']):
                ANCHORS = os.path.dirname(os.path.realpath(__file__)) + "/yolov4_anchors.txt"
                STRIDES = [8, 16, 32]
                XYSCALE = [1.2, 1.1, 1.05]

                ANCHORS = get_anchors(ANCHORS)
                STRIDES = np.array(STRIDES)

                detections = [np.expand_dims(out1, axis=0), np.expand_dims(out2, axis=0), np.expand_dims(out3, axis=0)]
                pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
                bboxes = postprocess_boxes(pred_bbox, orig_img_hw, 416, 0.25)
                bboxes = np.asarray(nms(bboxes, 0.213, method='nms'))

                max_num_bboxes_in_single_img = max(
                    len(bboxes), max_num_bboxes_in_single_img)

                if self.output_names['bboxes'] in request.requested_output_names():
                    if len(bboxes) > 0:
                        batch_out['bboxes'].append(bboxes[:, :5])
                    else:
                        batch_out['bboxes'].append(np.array([]))
                if self.output_names['labels'] in request.requested_output_names():
                    if len(bboxes) > 0:
                        batch_out['labels'].append(
                            [COCOLabels(idx).name.lower() for idx in bboxes[:, 5]])
                    else:
                        batch_out['labels'].append([])

            if max_num_bboxes_in_single_img == 0:
                # When no detected object at all in all imgs in the batch
                for idx, _ in enumerate(batch_out['bboxes']):
                    batch_out['bboxes'][idx] = [[-1, -1, -1, -1, -1]]
                for idx, _ in enumerate(batch_out['labels']):
                    batch_out['labels'][idx] = ["0"]
            else:
                # The output of all imgs must have the same size for Triton to be able to output a Tensor of type self.output_dtypes
                # Non-meaningful bounding boxes have coords [-1, -1, -1, -1, -1] and label '0'
                # Loop over images in batch
                for idx, out in enumerate(batch_out['bboxes']):
                    if len(out) < max_num_bboxes_in_single_img:
                        num_to_add = max_num_bboxes_in_single_img - len(out)
                        to_add = -np.ones((num_to_add, 5))
                        if len(out) == 0:
                            batch_out['bboxes'][idx] = to_add
                        else:
                            batch_out['bboxes'][idx] = np.vstack((out, to_add))

                # Loop over images in batch
                for idx, out in enumerate(batch_out['labels']):
                    if len(out) < max_num_bboxes_in_single_img:
                        num_to_add = max_num_bboxes_in_single_img - len(out)
                        to_add = ['0'] * num_to_add
                        if len(out) == 0:
                            batch_out['labels'][idx] = to_add
                        else:
                            batch_out['labels'][idx] = out + to_add

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses        