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
import inspect
import json
import logging
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

from FCS.fixed_coordinate_system import fixed_to_original_coordinate_system, reformat_homography_dict
from config import DrawingParams, MetaProcessingParams, LogicParams
from utils.utils_ import load_and_process, sort_meta_by_key
from utils.utils_pandas_df import read_multiindex_pd, from_dataframe_to_dict


class TrackVisualization:
    def __init__(self, video_path, path_to_homography_dict, files_dir, start, end, name, resize_constant):
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise Exception("Cannot open video at {!r}".format(video_path))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if path_to_homography_dict is not None:
            self.homography_dict = reformat_homography_dict(path_to_homography_dict)
        else:
            self.homography_dict = None
        self.files_dir = files_dir
        self.files_info = {}
        self.start = start
        self.end = end
        self.frame_counter = self.start
        self.save_dir = os.path.join(os.path.split(video_path)[0], os.path.split(video_path)[-1][:-4] + name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.resize_constant = resize_constant
        self.ids_counters = {}
        self.files_info_pairs = None
        logo = cv2.imread('Logo_aihunters.png')
        self.logo = cv2.resize(logo, (self.width // 10, self.height // 10))

    def load_info(self):
        files = os.listdir(self.files_dir)
        for file in files:
            file_path = os.path.join(args.files_dir, file)
            if file.endswith('json'):
                curr_file_info = load_and_process(file_path)
                curr_file_info = sort_meta_by_key(curr_file_info)
            elif file.endswith('.csv'):
                meta_pandas = read_multiindex_pd(file_path)
                meta_dict = from_dataframe_to_dict(meta_pandas)
                if self.homography_dict is not None:
                    curr_file_info = fixed_to_original_coordinate_system(meta_dict, self.homography_dict,
                                                                         int(os.environ.get(
                                                                             'fixed_coordinate_resize_h')),
                                                                         int(os.environ.get(
                                                                             'fixed_coordinate_resize_w')),
                                                                         self.height,
                                                                         self.width)
                else:
                    curr_file_info = meta_dict

            else:
                raise ValueError('Check you input file formats!')
            self.files_info[file] = curr_file_info
            self.ids_counters[file] = []

    def make_pairs(self):
        pairs = list(chunks(list(self.files_info.keys()), 2))
        self.files_info_pairs = [[{'name': key, 'info': self.files_info[key]} for key in pair] for pair in pairs]

    def visualisation_in_progress(self):
        self.load_info()
        self.make_pairs()
        self.draw_video()

    def get_final_frame(self, frame):
        final_frames = []
        for info_pair in self.files_info_pairs:
            assert len(info_pair) <= 2
            pair_frames = []
            for single_info in info_pair:
                frame_copy = frame.copy()
                frame_copy = self.draw_rect_and_text(frame_copy, single_info)
                pair_frames.append(frame_copy)
                del frame_copy
            if len(pair_frames) == 2:
                concatenated = np.concatenate(pair_frames, axis=1)
            else:
                concatenated = pair_frames[0]
            final_frames.append(concatenated)
        if len(final_frames) == 2:
            final_frame = np.concatenate(final_frames, axis=0)
        else:
            final_frame = final_frames[0]
        return final_frame

    def draw_video(self):
        assert len(self.files_info.items()) == 1 or len(self.files_info.items()) % 2 == 0
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.start))
        success = True

        while success:
            success, frame = self.capture.read()
            logging.debug('frame_counter {} '.format(self.frame_counter))
            if not success:
                break
            final_frame = self.get_final_frame(frame)
            if DrawingParams.draw_logo:
                final_frame = self.add_logo(final_frame)

            path_for_new_image = os.path.join(self.save_dir, '__{:06d}.png'.format(self.frame_counter))
            final_frame_resized = cv2.resize(final_frame,
                                             (int(final_frame.shape[1] / self.resize_constant),
                                              int(final_frame.shape[0] / self.resize_constant)))
            print(path_for_new_image)
            cv2.imwrite(path_for_new_image, final_frame_resized)
            self.frame_counter += 1
            if self.end is not None:
                if self.frame_counter >= self.end:
                    success = False

    def draw_counter(self, frame, counter):
        frame_clean = frame.copy()
        frame[0:100, self.width - 270:self.width, :] = (0, 0, 0)
        frame = create_border(frame)
        frame = cv2.putText(frame, 'IDs:',
                            org=(self.width - 250, 75),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=4, color=(255, 255, 255))
        frame = cv2.putText(frame, '{}'.format(str(counter)),
                            org=(self.width - 120, 75),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(0, 0, 139))
        img = cv2.addWeighted(frame_clean, DrawingParams.alpha_text_id, frame, 1 - DrawingParams.alpha_text_id, 0)
        del frame_clean
        return img

    def draw_file_name(self, frame, text):
        frame_clean = frame.copy()
        labelSize = cv2.getTextSize('{}'.format(text), cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        frame[0:labelSize[0][1] + 30, 0:labelSize[0][0] + 30, :] = (0, 0, 0)
        frame = cv2.putText(frame, '{}'.format(text),
                            org=(20, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=4, color=(240, 250, 230))
        img = cv2.addWeighted(frame_clean, DrawingParams.alpha_text, frame, 1 - DrawingParams.alpha_text, 0)
        del frame_clean
        return img

    def draw_rect_and_text(self, frame, meta):

        if self.frame_counter in meta['info']:
            frame = self.put_meta_on(frame, meta)
        if DrawingParams.draw_file_name:
            frame = self.draw_file_name(frame, meta['name'])
        if DrawingParams.draw_ids_counter:
            counter = len(np.unique(self.ids_counters[meta['name']]))
            frame = self.draw_counter(frame, counter)

        return frame

    def put_meta_on(self, img, meta):
        curr_frame_info = meta['info'][self.frame_counter]
        file_name = meta['name']
        img_clear = img.copy()
        for single_id_info in curr_frame_info:
            ind = int(single_id_info['index'])
            if ind not in MetaProcessingParams.false_indexes:  # do not want count "trash"
                curr_color = DrawingParams.rgb_colors_new[
                    ind if ind < len(DrawingParams.rgb_colors_new) else ind % len(DrawingParams.rgb_colors_new)]
                if int(ind) not in self.ids_counters[file_name]:
                    self.ids_counters[file_name].append(ind)
                if 'x1' in single_id_info:  # face case
                    img = put_face_meta_on(img, single_id_info, ind, curr_color)
                elif 'person' in single_id_info:  # pose case
                    if DrawingParams.full_pose:
                        img = draw_pose(img, single_id_info, ind, curr_color)
                    else:
                        img = put_pose_meta_on(img, single_id_info, ind, curr_color)

                else:
                    raise ValueError('check your meta')
        if DrawingParams.thickness == -1:
            img = cv2.addWeighted(img_clear, DrawingParams.alpha, img, 1 - DrawingParams.alpha,
                                  0, img)
        img = self.add_text(img, curr_frame_info, file_name)

        del img_clear
        return img

    def add_text(self, img, curr_frame_info, file_name):
        img_clear = img.copy()
        for single_id_info in curr_frame_info:
            ind = int(single_id_info['index'])
            indSize = cv2.getTextSize('{}'.format(ind), cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            if ind not in MetaProcessingParams.false_indexes:  # do not want count "trash"
                curr_color = DrawingParams.rgb_colors_new[
                    ind if ind < len(DrawingParams.rgb_colors_new) else ind % len(DrawingParams.rgb_colors_new)]
                if int(ind) not in self.ids_counters[file_name]:
                    self.ids_counters[file_name].append(ind)
                if 'person' in single_id_info:
                    if isinstance(single_id_info['person'], list):
                        person__ = single_id_info['person']
                    else:
                        person__ = json.loads(single_id_info['person'])
                    coords = (int(person__[1][0] + 50), int(person__[1][1] + 20))
                    img[coords[1] - indSize[0][1] - 10:coords[1] + 20, coords[0]:coords[0] + indSize[0][0] + 20,
                    :] = (
                        0, 0, 0)
                elif 'x1' in single_id_info:
                    coords = (int(single_id_info['x2']), int(single_id_info['y1']))
                    img[coords[1] - indSize[0][1] - 10:coords[1] + 20, coords[0]:coords[0] + indSize[0][0] + 20, :] = (
                        0, 0, 0)
                else:
                    raise ValueError('check you input ')
                img = cv2.putText(img, '{}'.format(str(ind)),
                                  org=(coords[0] + 10, coords[1] + 10),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3,
                                  color=curr_color)
        img = cv2.addWeighted(img_clear, DrawingParams.alpha_id, img, 1 - DrawingParams.alpha_id,
                              0, img)
        del img_clear
        return img

    def add_logo(self, img):
        img[-self.logo.shape[0]:, -self.logo.shape[1]:, :] = self.logo
        img = create_border(img)
        return img


def create_border(image_b):
    """
    create image border
    """
    pt1 = (0, 0)
    pt2 = (image_b.shape[1] - 1, image_b.shape[0] - 1)
    image_b = cv2.rectangle(image_b, pt1, pt2, color=[0, 0, 139], thickness=3, lineType=8, shift=0)
    return image_b


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def put_face_meta_on(img, curr_rect, index, color):
    img = cv2.rectangle(img,
                        (int(curr_rect["x1"]), int(curr_rect["y1"])),
                        (int(curr_rect["x2"]), int(curr_rect["y2"])),
                        color=color, thickness=DrawingParams.thickness)
    return img


def draw_pose(img, curr_info, index, color):
    if isinstance(curr_info['person'], list):
        person__ = curr_info['person']
    else:
        person__ = json.loads(curr_info['person'])
    for pair in DrawingParams.pose_pair_connections:
        if np.isscalar(person__[pair[0]]) or np.isscalar(person__[pair[1]]):
            continue
        if person__[pair[0]][2] <= DrawingParams.conf_threshold or person__[pair[1]][2] <= DrawingParams.conf_threshold:
            continue
        img = cv2.circle(img, (int(person__[pair[0]][0]), int(person__[pair[0]][1])), 20, color=color)
        img = cv2.line(img, (int(person__[pair[0]][0]), int(person__[pair[0]][1])),
                       (int(person__[pair[1]][0]), int(person__[pair[1]][1])), color=color, thickness=45)

    return img


def put_pose_meta_on(img, curr_info, index, color):
    if isinstance(curr_info['person'], list):
        person__ = curr_info['person']
    else:
        person__ = json.loads(curr_info['person'])
    if sum(person__[1]) > 0 and sum(person__[8]) > 0:
        img = cv2.line(img,
                       (int(person__[1][0]),
                        int(person__[1][1])),
                       (int(person__[8][0]),
                        int(person__[8][1])),
                       color=color, thickness=10)
    for pair in LogicParams.parts_.keys_to_use_for_estimation_pairs:
        if pair[0] in curr_info and not np.isnan(curr_info[pair[0]]):
            img = cv2.putText(img, '{}'.format(str(index)),
                              org=(int(person__[1][0]) + 20,
                                   int(person__[1][1]) + 20),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3,
                              color=color)
            break
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='custom arguments without using Sacred library ')
    parser.add_argument('--files_dir',
                        help="path to folder containing files we want to draw on. can containt 1, 2, or 4 files")
    parser.add_argument('--video_path')
    parser.add_argument('--path_to_homography_dict',
                        default=None)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--resize_constant', default=4, type=int)
    parser.add_argument('--name')

    args = parser.parse_args()

    vis = TrackVisualization(args.video_path, args.path_to_homography_dict, args.files_dir, args.start, args.end,
                             args.name, args.resize_constant)
    vis.visualisation_in_progress()
