import logging
import base64
import io
import time

from PIL import Image
from scipy.misc import imresize
from flask import Flask, request, jsonify
from flask_cors import CORS

import argparse
import distutils.util
import os
import sys
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # NOQA

import cv2

import torch

import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
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
        print('Called with args:')
        print(args)

        assert args.image_dir or args.images
        assert bool(args.image_dir) ^ bool(args.images)

        if args.dataset.startswith("coco"):
            dataset = datasets.get_coco_dataset()
            cfg.MODEL.NUM_CLASSES = len(dataset.classes)
        elif args.dataset.startswith("keypoints_coco"):
            dataset = datasets.get_coco_dataset()
            cfg.MODEL.NUM_CLASSES = 2
        else:
            raise ValueError(f'Unexpected dataset name: {args.dataset}')

        print('load cfg from file: {}'.format(args.cfg_file))
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
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(
                load_name, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        if args.load_detectron:
            print("loading detectron weights %s" % args.load_detectron)
            load_detectron_weight(maskRCNN, args.load_detectron)

        maskRCNN = mynn.DataParallel(
            maskRCNN, cpu_keywords=['im_info', 'roidb'],
            minibatch=True, device_ids=[0])  # only support single GPU

        maskRCNN.eval()
        if args.image_dir:
            imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
        else:
            imglist = args.images
        num_images = len(imglist)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for i in xrange(num_images):
            print('img', i)
            im = cv2.imread(imglist[i])
            assert im is not None

            timers = defaultdict(Timer)

            cls_boxes, cls_segms, cls_keyps = im_detect_all(
                maskRCNN, im, timers=timers)

            im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                args.output_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )

        if args.merge_pdfs and num_images > 1:
            merge_out_path = '{}/results.pdf'.format(args.output_dir)
            if os.path.exists(merge_out_path):
                os.remove(merge_out_path)
            command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                    merge_out_path)
            subprocess.call(command, shell=True)

    def infer(self, img):

        start_time = time.time()
        aspect_ratio = img.size[0] / img.size[1]
        img = self.transform(img)
        img = img.unsqueeze(0)

        data = {
            "A": img,
            "A_paths": "test.jpeg"
        }
        self.model.set_input(data)
        self.model.test()
        visuals = self.model.get_current_visuals()
        for label, im_data in visuals.items():
            if 'fake' not in label:
                continue
            im = tensor2im(im_data)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = Image.fromarray(im)

            with io.BytesIO() as buf:
                im.save(buf, format="jpeg")
                buf.seek(0)
                encoded_string = base64.b64encode(buf.read())
                encoded_result_image = (
                    b'data:image/jpeg;base64,' + encoded_string
                )
                logger.info("Infer time: {}".format(time.time() - start_time))
                return encoded_result_image


app = Flask(__name__)
CORS(app)
mask_rcnn_worker = MaskRCNWorker()


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
        result = cycle_gan_worker.infer(image)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
    return jsonify({'result': result.decode('utf-8')})


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
    app.run(host='0.0.0.0', port=8080)
