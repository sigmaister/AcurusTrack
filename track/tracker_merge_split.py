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

import utils.utils_ as util
import utils.utils_pandas_df as pdu
from config import AcceptanceParams
from track.tracker import AbstractTracker
from config import MetaProcessingParams
import random


class TrackerMergerSpliter(AbstractTracker):
    def __init__(self, data_processing, meta_partition, files_work):
        super(TrackerMergerSpliter, self).__init__(data_processing, meta_partition, files_work)
        self.moves_dict = {3: self.merge_move, 2: self.split_move}
        self.number_of_moves = 2
        self.n_mc = util.iterations_number_dependent_on_track(len(self.data_processing.pairs_to_consider),
                                                              number_of_moves=self.number_of_moves)

    def choose_move(self):
        self.chosen_move = random.choice([2, 3])

    def propose(self):
        self.choose_move()
        self.moves_dict[self.chosen_move]()

    def internal_loop(self):
        """ Algorithm loop inside the window """
        best_ratio = -np.inf
        while self.iteration < self.n_mc:
            self.iteration += 1
            self.complete_iter_number += 1
            self.accepted = False
            self.propose()  # make some movement
            logging.debug('change track {} '.format(self.change_track))
            if self.returned_state:  # no ability to make chosen move
                self.returned_state = False
                continue
            if self.proposed_partition is None:  # no ability to make chosen move
                continue
            self.acc_obj.propose(self.proposed_partition[['frame_no', 'id']].values,
                                 pdu.get_particular_states(self.proposed_partition,
                                                           self.change_track[
                                                               'new'])
                                 )

            accepted_count = self.acc_obj.analyse_acceptance(self.change_track)
            if accepted_count is None:
                continue
            ratio = max(list(self.acc_obj.ratio.values()))
            if ratio >= best_ratio:
                best_ratio = ratio
            if accepted_count >= AcceptanceParams.number_of_acc_for_acc:
                self.acc_obj.accepted_(change_track=self.change_track)
                self.acc_update()
                self.iteration = 0

    def split_move(self):
        # TODO of differemt from merge make as in merge_accelerated
        info = self.data_processing.main_info[
            ~self.data_processing.main_info['id'].isin(MetaProcessingParams.false_indexes)]
        info = info[info['len'] > 4]
        if info.empty:
            self.returned_state = True
            return
        info['chosen'] = info[info.columns[0:4]].apply(
            lambda x: [{'frame': i, 'index': x['id']}
                       for i in list(np.arange(x['start'] + 3, x['end'] - 2))],
            axis=1)  # TODO maybe remove len inside combinations
        c = info['chosen'].tolist()
        time_moments = [a for lst in c for a in lst]
        if not time_moments:
            self.returned_state = True
            return
        choose_time_moment = np.random.random_integers(
            0, len(time_moments) - 1)
        chosen_moment = time_moments[choose_time_moment]
        self.create_new_partition_split(chosen_moment)

    def create_new_partition_split(self, chosen_moment):
        new_df = self.data_processing.dataframe.copy(deep=True)
        new_df = pdu.change_index_in_df(new_df, chosen_moment['index'],
                                        max(self.data_processing.current_meta_indexes) + 1,
                                        end=chosen_moment['frame'] - 1)

        new_df = pdu.change_index_in_df(new_df, chosen_moment['index'],
                                        max(self.data_processing.current_meta_indexes) + 2,
                                        start=chosen_moment['frame'])
        self.change_track['current'] = [chosen_moment['index']]
        self.proposed_partition = new_df
        self.data_processing.df_grouped_ids_proposed= new_df.groupby([new_df.id])
        self.change_track['new'] = [
            max(self.data_processing.current_meta_indexes) + 1, max(self.data_processing.current_meta_indexes) + 2]
