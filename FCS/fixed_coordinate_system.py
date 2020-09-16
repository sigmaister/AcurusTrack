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

"""Functions regarding work with transition to absolute and relative coordinate systems"""

import json
from copy import deepcopy

import numpy as np
import numpy.linalg as linalg
import os

from config import LogicParams
import config
from config import MetaProcessingParams


def homography_transformation(coords_vector, matr_H, vector_len=3):
    if len(coords_vector) < vector_len:
        coords_vector = np.append(coords_vector, [1])
    new_vector = np.dot(matr_H, coords_vector)
    return new_vector[:-1] / new_vector[-1]


def inverse_homography_transformation(coords_vector, matr_H, vector_len=3):
    if len(coords_vector) < vector_len:
        coords_vector = np.append(coords_vector, [1])
    new_vector = np.dot(linalg.inv(matr_H), coords_vector)
    return new_vector[:-1] / new_vector[-1]


def to_fixed_coordinate_system(clean_meta, homography_dict,
                               fcs_height, fcs_width, img_height_orig,
                               img_width_orig):
    """Transform coordinates fo fixed coordinate system. In pose case remember about zero coordinates """
    recalculated_clean_meta = {}
    h_resize_coefficient = fcs_height / img_height_orig
    w_resize_coefficient = fcs_width / img_width_orig
    for frame_no, frame_info in clean_meta.items():
        recalculated_clean_meta[frame_no] = []
        for rect in frame_info:
            new_rect = {}
            # for pair_of_keys in config.keys_to_use_for_estimation_pairs:
            for pair_of_keys in LogicParams.parts_.keys_to_use_for_fcs:
                condition = (rect[pair_of_keys[0]] > 0 and rect[
                    pair_of_keys[1]] > 0) if LogicParams.face_or_pose_meta == 'pose' else True
                if condition:
                    [new_rect[pair_of_keys[0]],
                     new_rect[pair_of_keys[1]]] = np.around(homography_transformation(
                        [w_resize_coefficient * rect[pair_of_keys[0]],
                         h_resize_coefficient * rect[pair_of_keys[1]]],
                        homography_dict[frame_no]), decimals=2)
                    if len(list(new_rect.keys())) == 2:  # TODO WHY IS IT HERE
                        for additional_key in list(
                                set(list(rect.keys())).difference(set(LogicParams.parts_.keys_to_use_for_estimation))):
                            new_rect[additional_key] = rect[additional_key]
            if new_rect != {}:
                recalculated_clean_meta[frame_no].append(new_rect)
    return recalculated_clean_meta


def reformat_homography_dict(path_to_homographies):
    with open(path_to_homographies, "r") as curr_json:
        homography_matrices = json.load(curr_json)
        resize_info = homography_matrices["resize_info"]
        os.environ['fixed_coordinate_resize_h'] = str(resize_info["h"])
        os.environ['fixed_coordinate_resize_w'] = str(resize_info["w"])
        del homography_matrices["resize_info"]
        homography_dict = {int(k): v for k, v in homography_matrices.items()}
    reformat_homography_dict = {}
    next_H = None
    H_first = True
    # make identity matrix for first frame processing
    reformat_homography_dict[1] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for frame_no, frame_H in homography_dict.items():
        if frame_H["H"] is None:
            next_H = next_H
        else:
            if H_first:
                next_H = frame_H["H"]
                H_first = False
            else:
                next_H = np.dot(frame_H["H"], next_H)
        reformat_homography_dict[frame_no] = next_H

    reformat_homography_dict["resize_info"] = resize_info
    return reformat_homography_dict


def fixed_to_original_coordinate_system(meta,
                                        homography_dict,
                                        fcs_height, fcs_width,
                                        img_height_orig,
                                        img_width_orig):
    """ Return to original coordinate system.
    Coordinates may be different a little bit due to matrix calculation errors.

    """

    h_resize_coefficient = img_height_orig / fcs_height
    w_resize_coefficient = img_width_orig / fcs_width
    recalculated_final_meta = {}
    for frame_no, frame_info in meta.items():
        recalculated_final_meta[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            for pair_of_keys in LogicParams.parts_.keys_to_use_for_fcs:
                if pair_of_keys[0] in rect:
                    if np.isnan(rect[pair_of_keys[0]]):
                        continue
                    [new_rect[pair_of_keys[0]],
                     new_rect[pair_of_keys[1]]] = np.around(inverse_homography_transformation(
                        [rect[pair_of_keys[0]], rect[pair_of_keys[1]]],
                        homography_dict[frame_no]) * [h_resize_coefficient, w_resize_coefficient], decimals=2)

            recalculated_final_meta[frame_no].append(new_rect)
    return recalculated_final_meta
