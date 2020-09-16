"""
This file is part of AcurusTrack.

    AcurusTrack is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AcurusTrack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with AcurusTrack.  If not, see <https://www.gnu.org/licenses/>.
"""

import inspect
import json
import logging
import os
import shutil
import sys
from timeit import default_timer as timer

import cv2
import wget

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
import config
from FCS.fixed_coordinate_system import reformat_homography_dict
from pipeline import MainAlgo
from run import process_initial_dirs

logging.basicConfig(level=logging.INFO)

video_link = "https://drive.google.com/uc?export=download&id=1Ljfm7WgOD3M25kJOt-gCq-JC4tq5J8Lx"

video = os.path.join(currentdir, 'people_masks_occlusions_part_1.mp4')
video_name = 'people_masks_occlusions_part_1.mp4'
detections = os.path.join(currentdir, 'people_masks_occlusions_part_1.mp4_detections.json')
homography_dict_name = os.path.join(currentdir, 'dict_with_homography_matrix.json')
if not os.path.exists(video):
    wget.download(video_link, os.path.join(currentdir, video))

if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    py_cache_path = os.path.join(dir_path, '__pycache__')
    try:
        shutil.rmtree(py_cache_path)
    except BaseException:
        print('Error while deleting directory')

    exp_name = 'dema_people_in_masks'

    capture = cv2.VideoCapture(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.environ['img_w'] = str(width)
    os.environ['img_h'] = str(height)
    os.environ['VIDEO_NAME'] = video_name
    os.environ['exp_name'] = exp_name
    save_dir = os.path.join(currentdir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.environ['save_dir'] = save_dir

    assert config.AcceptanceParams.number_of_acc_for_acc == 1
    assert config.LogicParams.face_or_pose_meta == 'face'  # if not, change it in config.py

    if not os.path.exists(os.environ['save_dir']):
        os.makedirs(os.environ['save_dir'])
    timer0 = timer()
    homography_dict = reformat_homography_dict(homography_dict_name)
    with open(detections, 'r') as clean__:
        detections = json.load(clean__)
        detections = {int(k): v for k, v in detections.items()}
    process_initial_dirs(video_name, os.environ['save_dir'], exp_name)
    algorithm = MainAlgo(detections, homography_dict,
                         global_start_frame=1)
    algorithm.run_analyser()
    logging.info('all time {} '.format(timer() - timer0))
