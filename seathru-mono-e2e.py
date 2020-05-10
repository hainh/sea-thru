from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import time
import io

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import pynng
from pynng import nng

import torch
from torchvision import transforms, datasets

import deps.monodepth2.networks as networks
from deps.monodepth2.layers import disp_to_depth
from deps.monodepth2.utils import download_model_if_doesnt_exist

from email.mime.image import MIMEImage

from seathru import *
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from os.path import basename
import yaml

config = yaml.safe_load(open('configuration.yml'))
username = config['username']
password = config['password']

class Args:
    def __init__(self):
        self.f = 2.0
        self.l = 0.5
        self.p = 0.1
        self.min_depth = 0.1
        self.max_depth = 1.0
        self.size = 2048
        self.monodepth_add_depth = 1.0
        self.monodepth_multiply_depth = 10.0
        self.model_name = "mono_1024x320"
        self.no_cuda = True
        self.port = 8123
        self.output_graphs = False
        self.spread_data_fraction = 0.05

def send_mail(send_from: str, send_to: str, subject: str, text: str, img):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Subject'] = subject

    msg.attach(MIMEImage(img))

    smtp = smtplib.SMTP(host="smtp.gmail.com", port=587)
    smtp.starttls()
    smtp.login(username,password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

def server(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    addr = 'tcp://127.0.0.1:{}'.format(args.port)
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad(), nng.Rep0(listen=addr, recv_timeout=1000) as sock:
        print('Listening on {}...'.format(addr))
        while True:
            try:
                # Load image and preprocess
                msg = sock.recv()
                print('Recv image of length {}'.format(len(msg)), flush=True)
                img = pil.open(io.BytesIO(msg)).convert('RGB')
                sock.send(b'OK')
                original_width, original_height = img.size
                input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                print('Preprocessed image', flush=True)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
                            np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
                print("Processed image", flush=True)
                print('Loading image...', flush=True)
                depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth,
                                                            args.monodepth_multiply_depth)
                recovered = run_pipeline(np.array(img) / 255.0, depths, args)
                recovered = exposure.equalize_adapthist(scale(np.array(recovered)), clip_limit=0.03)
                sigma_est = estimate_sigma(recovered, multichannel=True, average_sigmas=True) / 10.0
                recovered = denoise_tv_chambolle(recovered, sigma_est, multichannel=True)
                im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
                imbytes = io.BytesIO()
                im.save(imbytes, format='png')
                img = imbytes.getvalue()
                print('Done.')

                send_mail(
                    'seathru.server@gmail.com',
                    'gibbyje2@gmail.com',
                    'Your image is ready!',
                    'See attached.',
                    img)

            except pynng.Timeout:
                time.sleep(0.1)
            except IOError as e:
                print(e)
                time.sleep(.1)
            except e:
                print(e)


if __name__ == '__main__':
    args = Args()
    server(args)