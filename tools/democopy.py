from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

sys.path.append(r"D:\\Users\\Frederic\\DokumenteDokumente\\pysot")
#print(sys.path)
import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from trackingmultiple import *
import time

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def controlloop():
    #for videoname, signal in videofilelist:
    startframe, stopframe, video_name =  ct.initialize(0)
    positionlist = ct.get_start_position()
    outputlist = []
    for index, position in enumerate(positionlist, 1):
        startposition = (position.xmin, position.ymin, position.xmax-position.xmin, position.ymax-position.ymin)
        position_x_y_w_h = [position.xmin, position.ymin, position.xmax, position.ymax]
        outputlist.append(main(index, len(positionlist), startframe, stopframe, startposition, position_x_y_w_h, video_name))
    with open("D:/Users/Frederic/DokumenteDokumente/INAVET/Versuch/boxes/versuch.txt", "w+") as boxesfile:
        outputstring = ''
        rows = len(outputlist[0])
        for i in range(0, rows):
            for positionlist in outputlist:
                outputstring += ','.join(positionlist[i]) + '-'
            outputstring += '\n'
        boxesfile.write(outputstring)

def main(signalindex, totalsignals, startframe, stopframe, startposition, position_x_y_w_h, video_name):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True

    """ if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN) """

    output = []
    #for frame in get_frames(args.video_name):
    f = startframe
    counter = 1
    while f < stopframe: # startframe + 5: #
        starttime = time.time()
        frame = ct.get_frame_from_index(f)
        if first_frame:
            try:
                init_rect = startposition # cv2.selectROI(video_name, frame, False, False)
                output.append([str(x) for x in position_x_y_w_h])
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 1)
                bboxi = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                bboxi = [str(x) for x in bboxi]
                output.append(bboxi)
            #cv2.imshow(video_name, frame)
            #cv2.waitKey(40)
        print("Signal:", signalindex, "of", totalsignals, "- Frame", counter, "of", stopframe-startframe, "calculated")
        f += 1
        counter += 1
    endtime = time.time()
    print("Speed:", counter / (endtime - starttime))
    return output


if __name__ == '__main__':
    ct = Controltool()
    controlloop()