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

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import DrawingParams, MetaProcessingParams, LogicParams


def frame_processing(frame_no_str, rects, first_frame_no=1,
                     draw_false_tracks=False):
    frame_no = int(frame_no_str)
    x_coords_list = []
    y_coords_list = []
    inds = []
    colors = []
    for rect in rects:
        if np.isnan(rect[LogicParams.parts_.keys_to_use_for_estimation[0]]):
            continue
        ind = int(rect['index'])
        if not draw_false_tracks:
            if ind in MetaProcessingParams.false_indexes:
                continue
        if LogicParams.parts_.keys_to_use_for_estimation[0] in rect:
            x_coords_list.append(
                int(os.environ.get('fixed_coordinate_resize_w')) * (frame_no - first_frame_no) + rect[
                    LogicParams.parts_.keys_to_use_for_estimation[0]])
            y_coords_list.append(rect[LogicParams.parts_.keys_to_use_for_estimation[1]])
        else:
            x_coords_list.append(
                int(os.environ.get('fixed_coordinate_resize_w')) * (frame_no - first_frame_no) + rect[
                    LogicParams.parts_.keys_to_use_for_estimation[0]])
            y_coords_list.append(rect[LogicParams.parts_.keys_to_use_for_estimation[1]])
        inds.append(ind)
        colors.append(DrawingParams.colors_names[ind % len(DrawingParams.colors_names)])
    return x_coords_list, y_coords_list, inds, colors


def draw_partitions_as_dots(dict_info, img_w, name, res_dir, inch_1=28.5, inch_2=10.5,
                            number_of_ticks=10, y_lim=1280):
    plt.clf()
    plt.ylim(0, y_lim)
    frames = list(map(int, list(dict_info.keys())))
    results = map(frame_processing, frames, list(dict_info.values()))

    all_coords = [i for i in results]
    x_list = [i[0] for i in all_coords]
    y_list = [i[1] for i in all_coords]
    indexes_unique = [i[2] for i in all_coords]
    colors = [i[3] for i in all_coords]
    x_list = [i for subl in x_list for i in subl]
    y_list = [i for subl in y_list for i in subl]
    colors = [i for subl in colors for i in subl]
    indexes_unique = [i for subl in indexes_unique for i in subl]

    colors_unique = np.unique(colors)
    plt.scatter(x_list, y_list, c=colors)
    if (len(x_list)) > 0 and (frames[-1] - frames[0]) // number_of_ticks > 0:
        plt.xticks(np.arange(0, len(frames) * img_w, len(frames) * img_w // number_of_ticks),
                   np.arange(frames[0], frames[-1], (frames[-1] - frames[0]) // number_of_ticks))

    recs = []
    for i in range(0, len(colors_unique)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors_unique[i]))
    plt.legend(recs, indexes_unique, loc=4)
    plt.title("track numbers {}".format(len(recs)))
    fig = plt.gcf()
    fig.set_size_inches(inch_1, inch_2)
    fig.savefig(os.path.join(res_dir, '{}_{}.png'.format(name, LogicParams.parts_.keys_to_use_for_estimation[0])))


def draw_partition(dict_info, img_w, name, res_dir, inch_1=28.5, inch_2=10.5,
                   number_of_ticks=20):
    if MetaProcessingParams.fixed_coordinate_system:
        draw_partitions_as_dots(dict_info, int(int(os.environ.get('fixed_coordinate_resize_w'))), name, res_dir,
                                inch_1=inch_1,
                                inch_2=inch_2,
                                number_of_ticks=number_of_ticks,
                                y_lim=int(os.environ.get('fixed_coordinate_resize_h')))
    else:
        draw_partitions_as_dots(dict_info, img_w, name, res_dir, inch_1=inch_1, inch_2=inch_2,
                                number_of_ticks=number_of_ticks, y_lim=int(os.environ.get('img_h')))
