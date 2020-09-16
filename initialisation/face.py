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

from copy import deepcopy

import numpy as np

from config import LogicParams
from initialisation.face_utils import iou_calc
from initialisation.initialisation import MetaInitialisation


class FaceMetaInitialisation(MetaInitialisation):
    def __init__(self, meta):
        MetaInitialisation.__init__(self, meta)
        self.iou_threshold = LogicParams.init_params.iou_threshold

    def looking_for_candidate_person(self):
        similarity = self.get_similarity()
        self.curr_person_distances[similarity] = self.prev_frame_person

    def get_similarity(self):
        similarity = iou_calc(self.current_frame_person, self.prev_frame_person)
        return similarity

    def analyse_distances(self, person):
        if self.curr_person_distances:
            max_sim = max(list(self.curr_person_distances.keys()))
        else:
            max_sim = 0

        if max_sim >= self.iou_threshold and self.curr_person_distances[max_sim]['index'] not in self.chosen_indexes:
            new_info = self.set_info(
                person,
                self.curr_person_distances[max_sim]['index'])
            self.persons_info.append(new_info)
            self.chosen_indexes.append(self.curr_person_distances[max_sim]['index'])
        else:
            new_info = self.set_info(
                person,
                self.ids_counter)
            self.persons_info.append(new_info)
            self.ids_counter += 1

    def set_info(self, person_full, curr_index):
        new_info = {key: value for key, value in person_full.items()}
        new_info['index'] = curr_index
        return new_info


def get_tracks(partition, end_frame_max=None):
    tracks_partition = {}

    for frame_no, frame_boxes in partition.items():
        if end_frame_max is not None:
            if int(frame_no) > end_frame_max:
                break  # do not want to track more than end
        for box in frame_boxes:
            if np.isnan(box['x1']):
                continue
            track_no = box['index']
            if track_no not in tracks_partition:
                tracks_partition[track_no] = []
            new_rect = {k: box[k] for k in box.keys()}
            new_rect["frame_no"] = frame_no
            if 'nature' in box:
                new_rect['nature'] = box['nature']
            if 'last_updated_by_detection' in box:
                new_rect['last_updated_by_detection'] = box['last_updated_by_detection']

            tracks_partition[track_no].append(new_rect)
    return tracks_partition


def from_tracks_to_partition(tracks):
    partition = {}
    for track_no, track_info in tracks.items():
        for rect in track_info:
            frame_no = rect['frame_no']
            if frame_no not in partition:
                partition[frame_no] = []
            new_rect = {k: rect[k] for k in rect.keys()}
            new_rect["index"] = track_no
            if 'nature' in rect:
                new_rect['nature'] = rect['nature']
            if 'last_updated_by_detection' in rect:
                new_rect['last_updated_by_detection'] = rect['last_updated_by_detection']

            partition[frame_no].append(new_rect)
    return partition


def add_centers_to_meta(meta):
    for frame_no, frame_info in meta.items():
        for rect in frame_info:
            if np.isnan(rect['x1']):
                continue
            rect['center_x'] = (rect['x1'] + rect['x2']) / 2
            rect['center_y'] = (rect['y1'] + rect['y2']) / 2
    return meta


def fill_in_gaps(meta, iou_threshold_to_make_meta, frames_gap, false_indexes, frames_gap_no_iou):
    new = {}
    tracks = get_tracks(meta)
    new_tracks = deepcopy(tracks)
    prev_frame = None
    prev_rect = None
    for track_no, rects in tracks.items():
        if track_no not in new:
            new[track_no] = []
        rects = sorted(rects, key=lambda k: k['frame_no'])
        if track_no not in false_indexes:
            for rect in rects:
                if prev_frame is None:
                    prev_frame = rect['frame_no']
                    prev_rect = rect
                    continue
                curr_gap = int(rect['frame_no']) - int(prev_frame)
                curr_iou = iou_calc(prev_rect, rect)
                if (
                        curr_iou >= iou_threshold_to_make_meta and 1 < curr_gap <= frames_gap) or curr_gap < frames_gap_no_iou:
                    for counter in range(1, int(rect['frame_no']) - int(prev_frame)):
                        new_rect = {'x1': (rect['x1'] - prev_rect['x1']) / curr_gap * counter + prev_rect['x1'],
                                    'x2': (rect['x2'] - prev_rect['x2']) / curr_gap * counter + prev_rect['x2'],
                                    'y1': (rect['y1'] - prev_rect['y1']) / curr_gap * counter + prev_rect['y1'],
                                    'y2': (rect['y2'] - prev_rect['y2']) / curr_gap * counter + prev_rect['y2'],
                                    "frame_no": counter + int(prev_frame), 'nature': 'approximation'}
                        new_tracks[track_no].append(new_rect)
                        new[track_no].append(new_rect)
                prev_frame = rect['frame_no']
                prev_rect = rect

    appr_partition = from_tracks_to_partition(new_tracks)
    appr_partition = add_centers_to_meta(appr_partition)
    return appr_partition
