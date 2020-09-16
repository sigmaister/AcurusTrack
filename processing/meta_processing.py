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
from timeit import default_timer as timer

import pandas as pd

import FCS.fixed_coordinate_system as fixu
import initialisation.face as facu
import initialisation.pose as posu
import utils.utils_pandas_df as pdu
from config import LogicParams, MetaProcessingParams


class MetaPartition:
    def __init__(self, partition, partition_df, homography_dict, name):
        self._observers = []
        self.name = name
        self.__homography_dict = homography_dict
        self.state = ['initial']
        self.__partition = partition
        self.__partition_df = partition_df
        self.info_name = 'init'

    def apply(self, meta_prep_object):
        meta_prep_object.apply(self)

    def add_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
        else:
            logging.info('observer is here already ')

    def remove_observer(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            logging.info('observer is not exists')

    def notify(self):
        logging.debug('notify')
        """ Update each observer"""
        for observer in self._observers:
            observer.notify(self)

    @property
    def homography(self):
        return self.__homography_dict

    @property
    def data(self):
        return self.__partition

    @data.setter
    def data(self, new_partition):
        try:
            self.__partition = new_partition
        except ValueError as e:
            logging.info('Error: {}'.format(e))
        else:
            logging.debug('data partititon setter')

    @property
    def data_df(self):
        return self.__partition_df

    @data_df.setter
    def data_df(self, new_partition):
        try:
            pdu.check_for_dublicates(new_partition)  # check for a duplicates
            self.__partition_df = new_partition
        except ValueError as e:
            logging.info('Error: {}'.format(e))
        else:
            logging.debug('data_df partition setter')
            self.notify()

    def print_pandas(self):
        """ Print pandas dataframe in "full" mode """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.data)


def fixed_coordinates(func):
    def decorator(ref, meta_obj):
        func(ref, meta_obj)
        if MetaProcessingParams.fixed_coordinate_system:
            logging.info('go to FCS')
            meta_obj.data = fixu.to_fixed_coordinate_system(
                meta_obj.data, meta_obj.homography, int(os.environ.get('fixed_coordinate_resize_h')),
                int(os.environ.get('fixed_coordinate_resize_w')), int(os.environ.get('img_h')),
                int(os.environ.get('img_w'))

            )

    return decorator


def add_keys(func):
    def decorator(ref, meta_obj):
        func(ref, meta_obj)
        if LogicParams.face_or_pose_meta == 'face':
            logging.info('add centers coordinates to faces meta')
            meta_obj.data = facu.add_centers_to_meta(meta_obj.data)

    return decorator


class MetaPreparation(object):
    """ Prepare meta for algorithm. Changes are possible. """

    def __init__(self):
        logging.debug('MetaPreparation init')

    def apply(self, meta_object):
        self.prepare_meta(meta_object)
        self.meta_ids_initialisation(meta_object)
        self.get_pandas_representation(meta_object)

    @fixed_coordinates
    @add_keys
    def prepare_meta(self, meta_object):
        """  transform it to FCS if needed"""
        meta_object.state.append('prepared')

    @staticmethod
    def meta_ids_initialisation(meta_object):
        """Initialise and save (Initialisation should be done in any case, so there are no any conditions)"""
        if LogicParams.face_or_pose_meta == 'pose':

            initialise_pose = posu.PoseMetaInitialisation(meta_object.data)
            initialised_meta = initialise_pose.initialisation()

        elif LogicParams.face_or_pose_meta == 'face':
            timer1 = timer()
            initialise_ = facu.FaceMetaInitialisation(meta_object.data)
            initialised_meta = initialise_.initialisation()
            appr_meta = facu.fill_in_gaps(initialised_meta, 0.3, 4, [-1, -10],
                                          2)
            initialise_ = facu.FaceMetaInitialisation(appr_meta)  # repeat initialisation after approximation
            initialised_meta = initialise_.initialisation()

            logging.debug('time {} '.format(timer() - timer1))

        else:
            raise ValueError('Check your processing type')

        meta_object.data = initialised_meta
        meta_object.state.append('initialised')

    @staticmethod
    def get_pandas_representation(meta_object):
        meta_object.data_df = pdu.dataframe_from_dict(meta_object.data)
        meta_object.state.append('got pandas')
