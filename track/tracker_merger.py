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


class TrackerMerger(AbstractTracker):
    def __init__(self, data_processing, meta_partition, files_work):
        super(TrackerMerger, self).__init__(data_processing, meta_partition, files_work)
        self.moves_dict = {3: self.merge_move}
        self.number_of_moves = 1
        self.n_mc = util.iterations_number_dependent_on_track(len(self.data_processing.pairs_to_consider),
                                                              number_of_moves=self.number_of_moves)

    def choose_move(self):
        self.chosen_move = 3

    def propose(self):
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
