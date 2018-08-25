import logging
import base64
import io
import time

from PIL import Image
from scipy.misc import imresize
from flask import Flask, request, jsonify
from flask_cors import CORS

from options.test_options import TestOptions
from data.base_dataset import get_transform
from models import create_model
from util.util import tensor2im

logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


class CycleGANWorker:

    def __init__(self):
        logger.info("Initializing ..")
        opt = TestOptions().parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        self.transform = get_transform(opt)

        self.model = create_model(opt)
        self.model.setup(opt)
        logger.info("Initialization done")

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
cycle_gan_worker = CycleGANWorker()


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify({"message": "Hi!"})


@app.route('/cyclegan', methods=['POST'])
def cyclegan():
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
