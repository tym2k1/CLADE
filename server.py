#!/usr/bin/env python3
"""
spade_server_direct.py

POST label+instance images to /infer. Preview last generated image at /.
No temp datasets, no dataloader. Minimal in-process SPADE/CLADE usage.
"""
import argparse
import threading
from pathlib import Path
from flask import Flask, request, send_file, render_template_string, jsonify
import torch
from PIL import Image
import numpy as np

app = Flask(__name__)
lock = threading.Lock()

HTML_PAGE = """<!doctype html>
<title>SPADE Preview</title>
<h1>Last generated image</h1>
{% if img_exists %}
  <img src="/last.png" style="max-width:90vw;max-height:80vh;border:1px solid #444"/>
{% else %}
  <p><i>No generated image yet. POST files (label, inst) to <code>/infer</code>.</i></p>
{% endif %}
"""

from options.base_options import BaseOptions

class Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='best', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')

        # parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(preprocess_mode='none',)
        # parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=512, load_size=512, display_winsize=512, z_dim=512)
        parser.set_defaults(name='coco')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1)
        parser.set_defaults(gpu_ids=-1)
        parser.set_defaults(how_many=1)
        parser.set_defaults(dataset_mode='coco')
        parser.set_defaults(checkpoints_dir='checkpoints')
        parser.set_defaults(dataroot='datasets/coco_stuff_populated')
        parser.set_defaults(norm_mode='clade')
        self.isTrain = False
        return parser

def build_model():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from util.util import tensor2im

    opt = Options().parse()

    model = Pix2PixModel(opt)
    model.eval()
    return model, tensor2im

@app.route('/')
def index():
    last_path = Path('last_generated.png')
    return render_template_string(HTML_PAGE, img_exists=last_path.exists())

@app.route('/last.png')
def last_png():
    last_path = Path('last_generated.png')
    if not last_path.exists():
        return 'No generated image yet', 404
    return send_file(last_path, mimetype='image/png')

@app.route('/infer', methods=['POST'])
def infer():
    if 'label' not in request.files or 'inst' not in request.files:
        return jsonify({'error': 'Please upload form fields label, inst'}), 400
    if not lock.acquire(blocking=False):
        return jsonify({'error': 'Server busy'}), 423

    try:
        label_nc = 183  # your model's label_nc
        label_f = request.files['label']
        inst_f = request.files['inst']

        # Convert images to integer tensors (C,H,W)
        label_img = Image.open(label_f).convert('L')
        inst_img = Image.open(inst_f).convert('L')

        label_np = np.array(label_img, dtype=np.int64)
        label_np[label_np >= label_nc] = 0

        label_tensor = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(1)  # [1,1,H,W]        
        inst_tensor  = torch.from_numpy(np.array(inst_img, dtype=np.int64)).unsqueeze(0).unsqueeze(1)

        H, W = label_tensor.shape[2:]  # height & width
        dummy_img = torch.zeros(1, 3, H, W, dtype=torch.float32)

        # Generate random latent z if requested
        z_dim = getattr(app.config['MODEL'].opt, 'z_dim', 256)
        # Accept a float POST field noise in [0,1], scale to randomness
        noise_scale = float(request.form.get('noise', 1.0))
        z = torch.randn(1, z_dim, dtype=torch.float32) * noise_scale
        
        data_dict = {
            'label': label_tensor,
            'instance': inst_tensor,
            'image': dummy_img,
            'dist': None, # or torch.zeros_like(label_tensor) if needed
            'z': z
        }

        model = app.config['MODEL']
        tensor2im = app.config['TENSOR2IM']

        device = torch.device('cpu')
        data_dict = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in data_dict.items()}

        with torch.no_grad():
            generated = model(data_dict, mode='inference')[0].detach().cpu()

        im_numpy = tensor2im(generated)
        Image.fromarray(im_numpy).save('last_generated.png')

        return jsonify({'status':'ok','generated':'last_generated.png'}), 200
    finally:
        lock.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    model, tensor2im = build_model()
    app.config['MODEL'] = model
    app.config['TENSOR2IM'] = tensor2im

    print(f"Server ready at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
