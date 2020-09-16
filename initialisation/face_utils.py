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

import math
from copy import deepcopy

import numpy as np

from config import FaceParams


def fill_and_format(observations, start_frame_segment, end_frame_segment):
    w = {}
    if observations == {}:
        return w
    key_from_meta = list(observations.keys())[0]
    change_from_str = False
    if type(key_from_meta) == str:
        change_from_str = True

    for i in range(int(start_frame_segment), int(end_frame_segment) + 1):
        index_to_find = i
        if change_from_str:
            index_to_find = str(i)
        if index_to_find in observations:
            w[i] = observations[index_to_find]
        else:
            w[i] = []  # fill in empty frames in json
    return w


def sort_meta_by_key(meta):
    new_dict = {}
    assert type(list(meta.keys())[0]) == np.int
    for key in sorted(meta.keys()):
        assert type(key) == np.int
        new_dict[key] = meta[key]
    return new_dict


def initialise_meta_by_iou(meta, iou_threshold_to_choose, gap, len_param=1):
    start = int(min(list(meta.keys())))
    end = int(max(list(meta.keys())))
    final_clean = fill_and_format(meta, start, end)
    meta = sort_meta_by_key(final_clean)

    find_from = deepcopy(meta)

    chains = []
    i = 0
    initial_points = {}
    while True:
        i += 1
        if not list(find_from.keys()):
            break
        frame_no = list(find_from.keys())[0]
        frame_info = find_from[frame_no]
        if not frame_info:
            del find_from[frame_no]
            if find_from == {}:
                break
            continue

        chosen = frame_info[0]
        initial_points[frame_no] = [chosen]
        chain, find_from = find_chain(find_from, chosen, start, end, iou_threshold_to_choose, frame_no, gap)
        if len(list(chain.keys())) == 1:
            chains.append(chain)
            continue

        if 0 < len(list(find_from.keys())) < 2:
            frame_no__ = list(find_from.keys())[0]
            chains.append({frame_no__: find_from[frame_no__]})
            del find_from[frame_no__]
            break

        if i > 0:
            chains.append(chain)

    chains_dict = reassign_colors_and_make_dict(chains, len_param)
    return chains_dict


def reassign_colors_and_make_dict(list_of_dicts, len_param):
    new_meta = {}
    for index, chain in enumerate(list_of_dicts):
        for frame_no, frame_info in chain.items():
            if not frame_info:
                continue

            if frame_no not in new_meta:
                new_meta[frame_no] = []
            assert len(frame_info) == 1
            rect = frame_info[0]
            rect_new = deepcopy(rect)
            if len(list(chain.keys())) <= len_param:
                new_index = -1
            else:
                new_index = index
            rect_new['index'] = new_index
            new_meta[frame_no].append(rect_new)
            del rect_new
    new_meta = {k: new_meta[k] for k in sorted(new_meta.keys())}
    return new_meta


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
                        new_rect = {key: (rect[key] - prev_rect[key]) / curr_gap * counter + prev_rect[key] for key in
                                    FaceParams.keys_pairs}
                        new_rect['frame_no'] = counter + int(prev_frame)
                        new_rect['nature'] = 'approximation'
                        new_tracks[track_no].append(new_rect)
                        new[track_no].append(new_rect)
                prev_frame = rect['frame_no']
                prev_rect = rect

    appr_partition = from_tracks_to_partition(new_tracks)
    return appr_partition


def swap(rect, key1, key2):
    if not (rect[key1] <= rect[key2]):
        temp = rect[key1]
        rect[key1] = rect[key2]
        rect[key2] = temp
    return rect


def swap_assigns(rect1, rect2):
    copy_rect1 = deepcopy(rect1)
    copy_rect2 = deepcopy(rect2)
    keys_pairs = [["x1", "x2"], ["y1", "y2"]]
    rects = {1: copy_rect1, 2: copy_rect2}
    new_rects = {1: copy_rect1, 2: copy_rect2}
    for name, rect in rects.items():
        for pair_of_keys in keys_pairs:
            new_rects[name] = swap(new_rects[name], pair_of_keys[0], pair_of_keys[1])

    return new_rects[1], new_rects[2]


def find_vertices(rect1, rect2):
    x1 = max(rect1['x1'], rect2['x1'])
    y1 = max(rect1['y1'], rect2['y1'])
    x2 = min(rect1['x2'], rect2['x2'])
    y2 = min(rect1['y2'], rect2['y2'])
    return x1, x2, y1, y2


def iou_calc(rect1, rect2, consider_inside=False):
    aligned_rect_1, aligned_rect_2 = swap_assigns(rect1, rect2)
    assert (aligned_rect_1["x1"] <= aligned_rect_1["x2"] and aligned_rect_1["y1"] <= aligned_rect_1["y2"]) or (
            aligned_rect_2["x1"] <= aligned_rect_2["x2"] and aligned_rect_2["y1"] <= aligned_rect_2[
        "y2"])
    x1, x2, y1, y2 = find_vertices(aligned_rect_1, aligned_rect_2)
    if x1 >= x2 or y1 >= y2:
        return 0
    intersection_area = math.fabs(x1 - x2) * math.fabs(y1 - y2)
    srect1 = math.fabs(aligned_rect_1['x1'] - aligned_rect_1['x2']) * math.fabs(
        aligned_rect_1['y1'] - aligned_rect_1['y2'])
    srect2 = math.fabs(aligned_rect_2['x1'] - aligned_rect_2['x2']) * math.fabs(
        aligned_rect_2['y1'] - aligned_rect_2['y2'])

    iou = float(intersection_area) / float((srect1 + srect2 - intersection_area))
    assert iou <= 1
    if consider_inside:
        if rect1['x1'] < rect2['x1'] and rect1['y1'] < rect2['y1'] and rect1['x2'] > rect2['x2'] and rect1['y2'] > \
                rect2['y2']:
            iou = 1.0
        if rect1['x1'] > rect2['x1'] and rect1['y1'] > rect2['y1'] and rect1['x2'] < rect2['x2'] and rect1['y2'] < \
                rect2['y2']:
            iou = 1.0
    return iou


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


def find_chain(set_to_find, initial_point, start, end, iou_threshold_to_choose, frame_no_, gap):
    chain = {frame_no_: [deepcopy(initial_point)]}
    set_to_find_new = deepcopy(set_to_find)
    last_to_compare = initial_point
    last_frame_to_compare = frame_no_
    for frame_no, frame_info in set_to_find.items():
        if frame_no < start or frame_info == [] or frame_no == frame_no_ or np.abs(
                frame_no - last_frame_to_compare) >= gap:
            continue
        if frame_no > end:
            break
        best_rect = None
        best_iou = -np.inf
        for index, rect in enumerate(frame_info):
            curr_iou = iou_calc(last_to_compare, rect)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_rect = deepcopy(rect)
        if best_rect is not None and best_iou >= iou_threshold_to_choose:
            chain[frame_no] = [best_rect]
            last_to_compare = best_rect
            last_frame_to_compare = frame_no
            set_to_find_new[frame_no].remove(best_rect)
            if not set_to_find_new[frame_no]:
                del set_to_find_new[frame_no]
    set_to_find_new[frame_no_].remove(initial_point)
    if not set_to_find_new[frame_no_]:
        del set_to_find_new[frame_no_]

    return chain, set_to_find_new


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
