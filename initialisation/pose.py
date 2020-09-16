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

import logging

import numpy as np

import utils.utils_math as um
from config import LogicParams
from initialisation.initialisation import MetaInitialisation


class PoseMetaInitialisation(MetaInitialisation):
    """ Meta initialisation for pose case"""

    def __init__(self, meta):
        filtered_meta = filter_meta(meta, LogicParams.init_params.confidence_score)
        logging.debug('POSE INIT')
        super(PoseMetaInitialisation, self).__init__(filtered_meta)
        self.curr_indexes_to_consider = LogicParams.init_params.curr_indexes_to_consider
        self.confidence_score = LogicParams.init_params.confidence_score
        self.len_pose_to_compare = LogicParams.init_params.len_pose_to_compare
        self.error_threshold = LogicParams.init_params.error_threshold

    def get_pose_similarity(self):

        curr_persons_errors_list = []
        assert len(self.current_frame_person['person']) == len(self.prev_frame_person['person'])
        for body_part_curr, body_part_prev in zip(self.current_frame_person['person'],
                                                  self.prev_frame_person['person']):
            # do not consider uncertain meta
            if body_part_curr[2] < self.confidence_score or body_part_prev[2] < self.confidence_score:
                continue

            curr_persons_errors_list.append(um.euclidean_norm_pose(body_part_prev,
                                                                   body_part_curr))
        return curr_persons_errors_list

    def looking_for_candidate_person(self):
        confident_meta = [i for i in self.prev_frame_person['person'] if i[2] > self.confidence_score]
        if len(confident_meta) < self.len_pose_to_compare:
            # if not enough data of which we are sure continue
            return
        self.curr_persons_errors[self.prev_frame_person['index']] = self.get_pose_similarity()
        if not self.curr_persons_errors[self.prev_frame_person['index']]:
            return
        metric = sum(self.curr_persons_errors[self.prev_frame_person['index']]) / len(
            self.curr_persons_errors[self.prev_frame_person['index']]) if len(
            self.curr_persons_errors[self.prev_frame_person[
                'index']]) > 0 else np.inf
        self.curr_person_distances[metric] = self.prev_frame_person

    def analyse_distances(self, person):
        keys_ = list(self.curr_person_distances.keys())
        if not keys_:
            # new_info = self.set_info(person,
            #                          self.ids_counter)
            # self.persons_info.append(new_info)
            # self.ids_counter += 1
            return
        else:
            min_dist = min(keys_)
        if min_dist <= self.error_threshold and self.curr_person_distances[min_dist][
            'index'] not in self.chosen_indexes:
            new_info = self.set_info(
                person,
                self.curr_person_distances[min_dist]['index'])
            self.persons_info.append(new_info)
            self.chosen_indexes.append(self.curr_person_distances[min_dist]['index'])
        else:
            new_info = self.set_info(person,
                                     self.ids_counter)
            self.persons_info.append(new_info)
            self.ids_counter += 1

    def set_info(self, person_full, curr_index):
        person = person_full['person']
        info = {'person': person, 'index': curr_index}
        for body_part_name, num_index in self.curr_indexes_to_consider.items():
            if num_index is None:
                # take mean
                info[body_part_name + '_x'] = (person[22][0] + person[19][0]) / 2
                info[body_part_name + '_y'] = (person[22][1] + person[19][1]) / 2
                info[body_part_name +
                     '_score'] = (person[22][2] + person[19][2]) / 2  # None means we take mean of the legs
            else:
                info[body_part_name + '_x'] = person[num_index][0]
                info[body_part_name + '_y'] = person[num_index][1]
                info[body_part_name + '_score'] = person[num_index][2]
        return info


def filter_meta(meta, confidence_score):
    filtered_meta = {}
    for frame_no, frame_items in meta.items():
        curr_frame_info = []
        for person in frame_items:
            if person['person'][8][2] >= confidence_score:  # consider confidence
                curr_frame_info.append(person)
        filtered_meta[frame_no] = frame_items
    return filtered_meta
