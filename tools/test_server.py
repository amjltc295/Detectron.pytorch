import logging
import time

from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
import jsonpickle

import argparse
import distutils.util
import sys
from collections import defaultdict

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # NOQA

import cv2

import torch

import _init_paths   # NOQA
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.net as net_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(
        description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA',
        action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


class MaskRCNNWorker:

    def __init__(self):

        """main function"""

        if not torch.cuda.is_available():
            sys.exit("Need a CUDA device to run the code.")

        args = parse_args()
        logger.info('Called with args:')
        logger.info(args)

        if args.dataset.startswith("coco"):
            self.dataset = datasets.get_coco_dataset()
            cfg.MODEL.NUM_CLASSES = len(self.dataset.classes)
        elif args.dataset.startswith("keypoints_coco"):
            self.dataset = datasets.get_coco_dataset()
            cfg.MODEL.NUM_CLASSES = 2
        else:
            raise ValueError(f'Unexpected dataset name: {args.dataset}')

        logger.info('load cfg from file: {}'.format(args.cfg_file))
        cfg_from_file(args.cfg_file)

        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)

        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            ('Exactly one of --load_ckpt and --load_detectron '
             'should be specified.')

        # Don't need to load imagenet pretrained weights
        cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
        assert_and_infer_cfg()

        maskRCNN = Generalized_RCNN()

        if args.cuda:
            maskRCNN.cuda()

        if args.load_ckpt:
            load_name = args.load_ckpt
            logger.info("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(
                load_name, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        if args.load_detectron:
            logger.info("loading detectron weights %s" % args.load_detectron)
            load_detectron_weight(maskRCNN, args.load_detectron)

        maskRCNN = mynn.DataParallel(
            maskRCNN, cpu_keywords=['im_info', 'roidb'],
            minibatch=True, device_ids=[0])  # only support single GPU

        maskRCNN.eval()

        self.maskRCNN = maskRCNN
        self.args = args

    def infer(self, img_PIL):
        start_time = time.time()

        im = np.array(img_PIL)[:, :, ::-1].copy()
        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(
            self.maskRCNN, im, timers=timers)

        logger.info("Infer time: {}".format(time.time() - start_time))
        result = {
            'boxes': jsonpickle.encode(cls_boxes),
            'segms': jsonpickle.encode(cls_segms),
            'keyps': jsonpickle.encode(cls_keyps)
        }
        return result


app = Flask(__name__)
mask_rcnn_worker = MaskRCNNWorker()


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify({"message": "Hi! This is a Mask RCNN worker."})


@app.route('/mask_rcnn', methods=['POST'])
def mask_rcnn():
    try:
        image_file = request.files['pic']
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no file['pic']"
        )
    if image_file is None:
        raise InvalidUsage('There is no iamge')
    try:
        image = Image.open(image_file)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['pic'] {request.files['pic']} "
            "could not be read by PIL"
        )
    try:
        result = mask_rcnn_worker.infer(image)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
        return jsonify({"result": result})


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
