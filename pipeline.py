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

import json
import multiprocessing
import os
import random

import numpy as np

import FCS.fixed_coordinate_system as fixu
import utils.processing.meta_processing_utils as mpu
import utils.utils_ as util
import utils.utils_pandas_df as pdu
from config import MetaProcessingParams, SystemParams, AcceptanceParams, LogicParams
from processing.dataframe_processing import DataframeProcessing
from processing.file_processing import FileProcessing
from processing.meta_processing import MetaPartition, MetaPreparation
from track.tracker_merger import TrackerMerger

"""Main components for running algorithm"""


def choose_best_csv_final_last(path_meta):
    folders_ = os.listdir(path_meta)
    folders = [f for f in folders_ if
               not (f.endswith('.png') or f.endswith('.сsv') or f.endswith('.txt')) and f != 'utils']
    folder = folders[0]
    final_folder_path = os.path.join(path_meta, folder)
    json_files = os.listdir(final_folder_path)
    best_json = [f for f in json_files if f.endswith("LAST_TRUE.сsv")]
    if not best_json:
        ratio_json = [f for f in json_files if f.endswith("BEST_RATIO.csv")]
        choice = ratio_json[0]
    else:
        choice = best_json[0]
    path_to_best_json = os.path.join(final_folder_path, choice)
    return path_to_best_json


def make_new_numeration_dict(func):
    def decorator(ref):
        meta, hom = func(ref)
        if MetaProcessingParams.renumbering:
            meta = mpu.change_meta_numeration(
                meta)
        return meta, hom

    return decorator


def get_original_coordinates(func):
    def decorator(ref):
        meta, homography_dict = func(ref)
        if MetaProcessingParams.fixed_coordinate_system:
            meta = fixu.fixed_to_original_coordinate_system(
                meta, homography_dict, int(os.environ.get('fixed_coordinate_resize_h')),
                int(os.environ.get('fixed_coordinate_resize_w')), int(os.environ.get('img_h')),
                int(os.environ.get('img_w'))
            )
        return meta, homography_dict

    return decorator


def make_new_numeration_pandas(func):
    def decorator(ref):
        meta = func(ref)
        if MetaProcessingParams.renumbering:
            new_indexes = pdu.get_update_indexes_rule(
                meta)  # renumbering by quantity
            meta = meta.replace({
                'id': new_indexes})
        return meta

    return decorator


class MainAlgo:
    def __init__(self, detections, homography_dict,
                 global_start_frame=None, global_end_frame=None):
        os.environ['PYTHONHASHSEED'] = str(SystemParams.seed)
        random.seed(SystemParams.seed)
        np.random.seed(SystemParams.seed)

        self.res_dir = os.environ.get('RES_DIR')
        self.homography = homography_dict

        self.start_frame, self.end_frame = self.determine_start_end(global_start_frame, global_end_frame, detections)
        self.full_meta = self.initialise_meta(detections, 'full')  # change detections inside
        self.__windows = self.get_windows()
        self.wind_objects = self.prepare_objs_for_each_window()

    @staticmethod
    def determine_start_end(start, end, detections):
        if not end:
            end = int(sorted(list(detections.keys()))[-1])
        else:
            a = detections.get(end, None)
            if not a:
                raise ValueError('there is no such end in meta')
        if not start:
            start = int(sorted(list(detections.keys()))[0])
        else:
            a = detections.get(start, None)
            if not a:
                raise ValueError('there is no such start in meta')
        return start, end

    @property
    def windows(self):
        return self.__windows

    @make_new_numeration_pandas
    def analysis(self):

        self.process_windows_separately()
        overlapped_windows = make_best_windows(
            self.res_dir)  # do merge by overlapped windows
        if len(self.wind_objects) > 1 and LogicParams.use_final_merge:
            final_meta = self.final_merge_single(overlapped_windows)
        else:
            final_meta = overlapped_windows

        return final_meta

    @make_new_numeration_dict
    @get_original_coordinates
    def get_meta(self):
        final_meta = self.analysis()
        final_meta_dict = pdu.from_dataframe_to_dict(
            final_meta)
        return final_meta_dict, self.homography

    def run_analyser(self):
        final_meta_dict, hom = self.get_meta()
        final_meta_dir = os.path.join(os.environ.get('EXP_DIR'), 'result')
        if not os.path.exists(final_meta_dir):
            os.makedirs(final_meta_dir)
        final_meta_path = os.path.join(final_meta_dir, 'result.json')
        with open(final_meta_path, 'w') as final_meta:
            json.dump(final_meta_dict, final_meta)

    def get_windows(self):
        """ Choose windows for processing according to density of the tracks"""
        ids = []
        curr_start = self.start_frame
        curr_window_len = 0
        windows = {}
        segment = {}
        frame_no = curr_start
        while True:
            if frame_no not in self.full_meta.data:
                frame_no += 1
                if frame_no == self.end_frame:
                    break
                continue
            frame_info = self.full_meta.data[frame_no]
            segment[frame_no] = frame_info
            for elem in frame_info:
                data = elem.get('index', None)
                if data is not None:
                    if elem['index'] not in ids:
                        ids.append(elem['index'])
                        curr_window_len += 1
            if curr_window_len > MetaProcessingParams.max_tracks_number_at_window or frame_no == self.end_frame:
                windows[str(curr_start) + '_' + str(frame_no)] = segment
                segment = {}
                curr_window_len = 0
                frame_no -= MetaProcessingParams.overlap
                curr_start = frame_no
                if frame_no + MetaProcessingParams.overlap >= self.end_frame:
                    break
            frame_no += 1
        assert windows
        return windows

    def prepare_objs_for_each_window(self):
        wind_objects = []
        for name, window in self.windows.items():
            meta_object = MetaPartition(window, pdu.dataframe_from_dict(window), self.homography, name)
            files_work = FileProcessing(meta_object, '{}'.format(name))
            meta_object.add_observer(files_work)
            processed_meta = DataframeProcessing(meta_object)
            meta_object.add_observer(processed_meta)
            tracker_obj = TrackerMerger(processed_meta, meta_object, files_work)
            # tracker_obj = TrackerMergerSpliter(processed_meta, meta_object, files_work)
            wind_objects.append(tracker_obj)
        return wind_objects

    def initialise_meta(self, meta, name):
        meta_start_end = util.fill_and_format(meta, self.start_frame, self.end_frame)
        meta_object = MetaPartition(meta_start_end, None, self.homography,
                                    name)  # do not have dataframe form, so pass None
        preparation = MetaPreparation()
        meta_object.apply(preparation)
        with open(os.path.join(self.res_dir, 'initialised.json'), 'w') as json_to_save:
            json.dump(meta_object.data, json_to_save)
        meta_object.data_df.to_csv(os.path.join(self.res_dir,
                                                '{}.csv'.format('initialised')))

        return meta_object

    def process_windows_separately(self):
        """ Process in parallel windows with algorithm"""
        if not SystemParams.use_multiprocessing:
            for single_obj in self.wind_objects:
                window_processing(single_obj)
        else:
            num_cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=num_cores)
            self.wind_objects = [(i, None) for i in self.wind_objects]
            pool.starmap(window_processing, self.wind_objects)
            pool.close()

    @staticmethod
    def update_config_for_final():
        AcceptanceParams.acc = 0.0
        os.environ['experiment_name_final'] = os.environ.get('exp_name') + '_merged_processed_final'
        os.environ['RES_DIR'] = os.path.join(
            os.environ.get('EXP_DIR'),
            os.environ.get('experiment_name_final'))
        if not os.path.exists(os.environ.get('RES_DIR')):
            os.makedirs(os.environ.get('RES_DIR'))

    def final_merge_single(self, overlapped_windows):
        """ Final merge, optional"""
        self.update_config_for_final()
        name = 'final'
        meta_object = MetaPartition(None, overlapped_windows, self.homography, name)
        file_dir = FileProcessing(meta_object, name)
        file_dir.create_dir()
        meta_object.add_observer(file_dir)
        processed_meta = DataframeProcessing(meta_object)
        meta_object.add_observer(processed_meta)
        tracker_obj = TrackerMerger(processed_meta, meta_object, file_dir)

        window_processing(tracker_obj, final_merge=True)
        final_json = util.choose_best_csv_final_last(os.environ.get('RES_DIR'))
        final_json_meta = pdu.read_multiindex_pd(final_json)
        return final_json_meta


def load_and_clean_csv(csv_path):
    curr_meta = pdu.read_multiindex_pd(csv_path)
    curr_meta = curr_meta[~curr_meta['id'].isin(MetaProcessingParams.false_indexes)]
    new_indexes = pdu.get_update_indexes_rule(curr_meta)
    curr_meta = curr_meta.replace({'id': new_indexes})
    curr_meta.to_csv(csv_path)
    return curr_meta


def make_best_windows(path_to_meta_folder):
    chosen_files = util.choose_csv_from_dir(path_to_meta_folder)
    curr_meta = load_and_clean_csv(chosen_files[0])
    all_info = curr_meta
    indexes_curr = pdu.get_current_meta_indexes(curr_meta)
    counter_curr = len(indexes_curr) + 1

    for i in range(1, len(chosen_files)):
        next_json_info = load_and_clean_csv(chosen_files[i])
        all_info, counter_curr = pdu.merge_two_consecutive_windows(all_info, next_json_info,
                                                                   counter_curr)
    all_info.to_csv(
        os.path.join(
            path_to_meta_folder,
            'final_processing_merged_MCMC.csv'))
    return all_info


def window_processing(wind_obj, final_merge=None):
    """ Processing of single window."""

    wind_obj.final_merge = final_merge
    wind_obj.algo_iteration()
