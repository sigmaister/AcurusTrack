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

import argparse
import json
import logging
import os
import shutil
from timeit import default_timer as timer

import cv2

from FCS.fixed_coordinate_system import reformat_homography_dict
from pipeline import MainAlgo
from config import MetaProcessingParams


# logging.basicConfig(level=logging.DEBUG)


def process_initial_dirs(video_name, save_dir, experiment_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_dir = os.path.join(exp_dir, experiment_name)
    os.environ['EXP_DIR'] = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    os.environ['RES_DIR'] = os.path.join(exp_dir, experiment_name)
    if not os.path.exists(os.environ['RES_DIR']):
        os.makedirs(os.environ['RES_DIR'])


def main(arguments):
    capture = cv2.VideoCapture(arguments.video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.environ['img_w'] = str(width)
    os.environ['img_h'] = str(height)
    os.environ['VIDEO_NAME'] = arguments.video_name
    os.environ['exp_name'] = arguments.exp_name
    if arguments.save_dir is not None:
        os.environ['save_dir'] = arguments.save_dir
    else:
        os.environ['save_dir'] = 'results'
    if not os.path.exists(os.environ['save_dir']):
        os.makedirs(os.environ['save_dir'])

    timer0 = timer()

    if arguments.path_to_homography_dict is not None:
        homography_dict = reformat_homography_dict(
            arguments.path_to_homography_dict)
    else:
        homography_dict = None
        assert not MetaProcessingParams.fixed_coordinate_system, 'If you do not want use fixed coordinates, make this parameter False'
    with open(arguments.detections, 'r') as clean__:
        detections = json.load(clean__)
        detections = {int(k): v for k, v in detections.items()}
    process_initial_dirs(arguments.video_name, os.environ['save_dir'], arguments.exp_name)

    logging.basicConfig(filename=os.path.join(os.environ['RES_DIR'], 'info.log'), level=logging.DEBUG)
    algorithm = MainAlgo(detections, homography_dict,
                         global_start_frame=arguments.start_frame, global_end_frame=arguments.end_frame)
    algorithm.run_analyser()
    logging.info('all time {} '.format(timer() - timer0))


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    py_cache_path = os.path.join(dir_path, '__pycache__')
    try:
        shutil.rmtree(py_cache_path)
    except BaseException:
        print('Error while deleting directory')

    parser = argparse.ArgumentParser(
        description='custom arguments without using Sacred library ')

    parser.add_argument('--detections')
    parser.add_argument('--video_path')
    parser.add_argument('--video_name')
    parser.add_argument('--exp_name')
    parser.add_argument('--path_to_homography_dict')
    parser.add_argument('--start_frame', default=1, type=int)
    parser.add_argument('--save_dir')
    parser.add_argument('--end_frame', default=None, type=int)

    args = parser.parse_args()
    main(args)
