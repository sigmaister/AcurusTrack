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
import logging
import os

import utils.utils_pandas_df as pdu
import visualization.visualization as visu
from config import DrawingParams


class FileProcessing:
    def __init__(self, meta, name):
        logging.debug('File processing init')
        self.__partition_dict = meta.data
        self.__partition_df = meta.data_df
        self.name = name
        self.res_dir = os.environ.get('RES_DIR')
        self.curr_window_dir = self.create_dir()
        if self.__partition_df is not None:
            self.write_df_to_csv(self.name, 'LAST_TRUE')

    @property
    def meta(self):
        return self.__partition_dict

    @meta.setter
    def meta(self, new_meta):
        self.__partition_dict = new_meta

    @property
    def meta_df(self):
        return self.__partition_df

    @meta_df.setter
    def meta_df(self, new_meta):
        self.__partition_df = new_meta

    def write_df_to_csv(self, file_name, name):
        assert self.meta_df is not None
        filename = os.path.join(self.curr_window_dir,
                                '_{}_{}.csv'.format(file_name, name))
        self.meta_df.to_csv(filename)
        logging.debug('{} file has been written '.format(filename))

    def notify(self, meta):
        self.meta_df = meta.data_df
        self.write_df_to_csv(self.name, 'LAST_TRUE')  # Save new meta when updated
        if DrawingParams.draw_accepted:
            dict_format = pdu.from_dataframe_to_dict(self.meta_df)
            visu.draw_partition(dict_format, int(os.environ['img_w']), meta.info_name,
                                self.curr_window_dir)

    def save_to_json(self, data_to_save, name):
        with open(os.path.join(self.res_dir, name + '.json'), 'w') as json_to_save:
            json.dump(data_to_save, json_to_save)

    def create_dir(self):
        curr_window_dir = os.path.join(
            self.res_dir, '_{}'.format(self.name))
        if not os.path.exists(curr_window_dir):
            os.makedirs(curr_window_dir)

        return curr_window_dir
