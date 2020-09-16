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
import os
from itertools import combinations

import numpy as np

import utils.utils_pandas_df as pdu
from config import MetaProcessingParams, LogicParams


class DataframeProcessing:
    """ No changes to original partition, just extract useful information about it."""

    def __init__(self, meta_class):
        self.meta_class = meta_class
        self.frames_no = {}
        self.__dataframe = meta_class.data_df
        self.df_grouped_ids = self.dataframe.groupby([self.dataframe.id])
        self.main_info = self.get_df_info()
        self.clean_meta()
        meta_class.data_df = self.__dataframe  # after cleaning notify and save cleaned partition into file
        self.df_grouped_ids = self.dataframe.groupby([self.dataframe.id])
        self.main_info = self.get_df_info()

        self.current_meta_indexes = pdu.get_current_meta_indexes(self.dataframe)
        self.states = self.df_grouped_ids.apply(pdu.get_tracks,
                                                LogicParams.parts_.keys_to_use_for_estimation_pairs).to_dict()

        self.pairs_to_consider = self.choose_non_overlapped_pairs()

    @property
    def dataframe(self):
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, new_dataframe):
        self.__dataframe = new_dataframe
        self.df_grouped_ids = new_dataframe.groupby([new_dataframe.id])
        self.main_info = self.get_df_info()
        logging.debug('notify')
        self.current_meta_indexes = pdu.get_current_meta_indexes(new_dataframe)
        self.frames_no = {}
        self.df_grouped_ids.apply(
            pdu.get_frames_numbers, global_dict=self.frames_no)
        for key in MetaProcessingParams.false_indexes:
            if key in self.frames_no.keys():
                self.frames_no.pop(key)
        self.states = self.df_grouped_ids.apply(pdu.get_tracks,
                                                LogicParams.parts_.keys_to_use_for_estimation_pairs).to_dict()
        self.pairs_to_consider = self.choose_non_overlapped_pairs()

    def clean_meta(self):
        """
        Find "trash" short detections, remove it
        :return:clean data without trash detections
        """

        dataframe_with_trash = self.find_trash_detections()
        dataframe_cleaned = self.clean_data(dataframe_with_trash)
        self.dataframe = dataframe_cleaned  # should call all necessary updates

    def get_df_info(self):
        """ Format meta and get info about single tracks"""
        start = self.df_grouped_ids.apply(pdu.starts)
        end = self.df_grouped_ids.apply(pdu.ends)
        start = start.reset_index(level=1, drop=True)
        info = start.copy()
        end = end.reset_index(level=1, drop=True)
        end['unique_id'] = end.groupby('id').cumcount()
        end.set_index('unique_id', append=True, inplace=True)
        end = end.reset_index(level=1, drop=True)
        info['end'] = end['frame_no'].values
        info.rename(columns={'frame_no': 'start'}, inplace=True)
        info['len'] = info['end'] - info['start']
        info['id'] = end.index.get_level_values(0)
        return info.copy(deep=True)

    def find_trash_detections(self):
        """
        Find tracks with some particular length, assign them "false indexes"
        :return: meta with false indexes
        """
        range_tracks_make_trash = self.main_info[self.main_info.len <=
                                                 MetaProcessingParams.len_to_make_trash_index]
        start__ = range_tracks_make_trash.start.values
        end__ = range_tracks_make_trash.end.values
        ids = range_tracks_make_trash.id.values
        dataframe_with_trash_detections = self.dataframe
        for start_, end_, id_ in zip(start__, end__,
                                     ids):
            dataframe_with_trash_detections = pdu.change_index_in_df(
                self.dataframe, id_, MetaProcessingParams.false_indexes[0], start_, end_)

        return dataframe_with_trash_detections

    @staticmethod
    def clean_data(data):
        """ Remove trash detections from data"""
        clean_data = data[~data['id'].isin(
            MetaProcessingParams.false_indexes)]
        return clean_data

    def notify(self, meta_class):
        """ Update information """
        self.dataframe = meta_class.data_df
        if not os.environ.get("FINAL_MERGE"):
            self.main_info = self.get_df_info()

        self.df_grouped_ids = self.dataframe.groupby([self.dataframe.id])
        self.current_meta_indexes = pdu.get_current_meta_indexes(self.dataframe)
        self.frames_no = {}
        self.df_grouped_ids.apply(
            pdu.get_frames_numbers, global_dict=self.frames_no)
        for key in MetaProcessingParams.false_indexes:
            if key in self.frames_no.keys():
                self.frames_no.pop(key)
        self.states = self.df_grouped_ids.apply(pdu.get_tracks,
                                                LogicParams.parts_.keys_to_use_for_estimation_pairs).to_dict()

        self.pairs_to_consider = self.choose_non_overlapped_pairs()

    def check_intersection_dict(self, pair):
        return set(self.frames_no[pair[0]]).intersection(
            self.frames_no[pair[1]]) == set()

    def choose_non_overlapped_pairs(self):
        ids_combinations = list(combinations(self.frames_no.keys(), 2))
        intersect_or_not = np.array(
            list(map(self.check_intersection_dict, ids_combinations)))
        chosen_pairs = []
        if intersect_or_not != []:
            chosen_pairs = list(
                np.array(ids_combinations)[
                    intersect_or_not])

        return chosen_pairs
