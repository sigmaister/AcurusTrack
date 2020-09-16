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

import numpy as np

from config import LogicParams, MetaProcessingParams


def make_new_numeration(all_indexes, false_indexes, meta):
    """

    :param all_indexes: indexes we replace
    :param false_indexes: indexes we do not change
    :param meta: data
    :return: meta with indexes 0,1,2,3,...
    """
    new_keys = {}
    counter = 0
    for key_ in all_indexes:
        if key_ in false_indexes:
            new_keys[key_] = key_
            continue
        new_keys[key_] = counter
        counter += 1

    for frame_no, frame_info in meta.items():
        for single_rect in frame_info:
            if single_rect['index'] in false_indexes:
                continue
            single_rect['index'] = new_keys[single_rect['index']] + 1000

    for frame_no, frame_info in meta.items():
        for single_rect in frame_info:
            if single_rect['index'] in false_indexes:
                continue
            single_rect['index'] = single_rect['index'] % 1000
    return meta


def change_meta_numeration(meta):
    all_indexes = []
    false_indexes = MetaProcessingParams.false_indexes

    tracks = get_tracks(meta)

    for track_no in tracks:
        if track_no not in all_indexes:
            all_indexes.append(track_no)

    cleaned_partition = make_new_numeration(all_indexes, false_indexes, meta)
    return cleaned_partition


def get_tracks(partition, end_frame_max=None):
    """

    :param partition: meta we work with in the dictionary form
    :param end_frame_max:maximal frame number
    :return: meta, united by tracks, not by frame numbers
    """
    tracks_partition = {}
    for frame_no, frame_boxes in partition.items():
        if end_frame_max is not None:
            if int(frame_no) > end_frame_max:
                break
        for box in frame_boxes:
            if np.all(np.isnan(np.array(
                    [box[k_] for k_ in LogicParams.parts_.keys_to_use_for_estimation]))):
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
