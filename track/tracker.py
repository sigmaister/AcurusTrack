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
import copy
import logging
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import os

import initialisation.pose_utils as posu
import utils.utils_ as util
import utils.utils_pandas_df as pdu
import visualization.visualization as visu
from additional.kalman_filter import KalmanFilter
from config import AcceptanceParams, LogicParams, DrawingParams, MetaProcessingParams
from config import KalmanParams


class AbstractTracker(ABC):
    @abstractmethod
    def __init__(self, data_processing, meta_partition, files_work):
        self.files_ = files_work
        self.final_merge = None
        self.p_d = AcceptanceParams.p_d
        self.p_z = AcceptanceParams.p_z
        self.lambda_b = AcceptanceParams.lambda_b
        self.lambda_f = AcceptanceParams.lambda_f
        self.change_track = {}
        self.chosen_move = 3
        self.acc_list = []
        self.u_list = []
        self.accepted = False
        self.u_random_curr_iter = None
        self.curr_acceptance = None
        self.ratio = None
        self.priors_parameters = {}
        self.acceptance = None
        self.__current_state = None
        self.meta_partition = meta_partition
        self.data_processing = data_processing
        self.cur_iter_name = 'initial'
        self.likelihood = None
        self.likelihoods = None
        self.priors = None
        self.first_iteration_done = False
        self.returned_state = False
        self.proposed_partition = None
        self.df_grouped_ids_proposed = None
        self.acc_obj = Acceptance(data_processing.dataframe[['frame_no', 'id']].values, data_processing.states)
        self.iteration = 0
        self.complete_iter_number = 0
        self.accepted = False

    def algo_iteration(self):
        """ Single interation of the algorithm"""
        if self.final_merge is None:
            self.internal_loop()
        else:
            self.final_merge_loop()

    def acc_update(self):
        self.accepted = True
        self.meta_partition.data_df = self.proposed_partition
        self.data_processing.dataframe = self.proposed_partition
        self.proposed_partition = None
        iter_info = round(self.acc_obj.u_random_curr_iter,
                          2) if self.acc_obj.u_random_curr_iter is not None else self.cur_iter_name
        self.meta_partition.info_name = str(self.complete_iter_number) + '_' + str(self.chosen_move) + '_' + str(round(
            max(list(self.acc_obj.curr_acceptance)),
            2)) + '_' + str(iter_info) + '_' + str(round(
            max(list(self.acc_obj.ratio.values())),
            2))

    @abstractmethod
    def internal_loop(self):
        raise NotImplementedError("Must override internal_loop")

    @abstractmethod
    def choose_move(self):
        raise NotImplementedError("Must override choose_move")

    @abstractmethod
    def propose(self):
        raise NotImplementedError("Must override propose")

    def final_merge_loop(self):
        flag = True
        check_ = 0
        counter = 0
        if not self.data_processing.pairs_to_consider:
            return
        while flag:
            counter += 1
            break_to_while = False
            if not self.data_processing.pairs_to_consider:
                break

            for index, pair in enumerate(self.data_processing.pairs_to_consider):
                self.cur_iter_name = str(counter) + '_' + str(pair)
                logging.debug('pair {} '.format(pair))
                if break_to_while:
                    check_ = 0
                    break
                self.accepted = False
                self.merge_move(final_merge=True,
                                pair=pair)
                if self.proposed_partition is None:
                    if check_ >= len(self.data_processing.pairs_to_considercleaned) ** 2:
                        break_to_while = False
                        flag = False
                        break
                    else:
                        check_ += 1
                        continue
                self.acc_obj.propose(self.proposed_partition[['frame_no', 'id']].values,
                                     pdu.get_particular_states(self.proposed_partition,
                                                               self.change_track[
                                                                   'new']))

                accepted_count = self.acc_obj.analyse_acceptance(self.change_track)
                ratio = max(list(
                    self.acc_obj.ratio.values()))
                if DrawingParams.draw_every_iteration:
                    current = pdu.from_dataframe_to_dict(self.meta_partition.data_df)
                    visu.draw_partition(current, int(os.environ.get('img_w')),
                                        'partitions_iteration_{}_{}_'.format(self.cur_iter_name,
                                                                             "current_"),
                                        self.files_.curr_window_dir)

                if accepted_count >= AcceptanceParams.number_of_acc_for_acc:
                    self.acc_obj.accepted_(change_track=self.change_track)
                    self.meta_partition.data_df = self.proposed_partition
                    self.acc_update()
                    break_to_while = True
                if break_to_while:
                    check_ = 0
                    break
                check_ += 1
                if not break_to_while and check_ >= len(self.data_processing.pairs_to_consider):
                    flag = False

    def merge_move(self, final_merge=None, pair=None):
        if final_merge is None:
            if not self.data_processing.pairs_to_consider:
                self.returned_state = True
                return

            pair_selection = np.random.random_integers(
                0, len(self.data_processing.pairs_to_consider) - 1)
            self.create_new_partition_merge(self.data_processing.pairs_to_consider[pair_selection])
            del self.data_processing.pairs_to_consider[pair_selection]
        else:
            self.create_new_partition_merge(pair)

    def create_new_partition_merge(self, pair_chosen):
        self.change_track['current'] = [pair_chosen[0], pair_chosen[1]]
        new_df = self.data_processing.dataframe.copy(deep=True)
        new_df = pdu.change_index_in_df(new_df, pair_chosen[0], max(
            self.data_processing.current_meta_indexes) + 1)
        new_df = pdu.change_index_in_df(new_df, pair_chosen[1], max(self.data_processing.current_meta_indexes) + 1)

        self.proposed_partition = new_df
        self.df_grouped_ids_proposed = new_df.groupby([new_df.id])
        self.change_track['new'] = [max(self.data_processing.current_meta_indexes) + 1]


class Acceptance:
    def __init__(self, frame_no_ind, states_):
        """

        :param frame_no_ind: list of lists in the form [[frame_no, ind], [frame_no, ind], ...] - information for priors computation
        :param states_: dict in the form {id : {('body_part_x','body_part_y'):[[x_1, x_2, ...],[y_1, y_2, ...]]}, {...}, ...} - information for likelihoods computation
        """
        self.first_iteration_done = False
        self.u_random_curr_iter = np.random.random()
        self.curr_acceptance = 0
        self.ratio = {}
        self.curr_priors_obj = Priors(frame_no_ind)
        self.curr_liks_obj = Likelihoods(states_)
        self.acceptance = {}
        self.proposed_priors_obj = None
        self.proposed_liks_obj = None

    def propose(self, frame_no_ind, states):
        self.u_random_curr_iter = np.random.random()
        self.proposed_priors_obj = Priors(frame_no_ind)
        self.proposed_liks_obj = Likelihoods(states)

    def analyse_acceptance(self, change_track):
        """ Analyzing ratio and acceptance. """

        self.curr_acceptance = self.get_acceptance(change_track)
        logging.debug('acc {} '.format(self.curr_acceptance))
        logging.debug(
            'self.curr_acceptance {} '.format(
                self.curr_acceptance))
        self.curr_acceptance = list(self.curr_acceptance.values())
        if not self.curr_acceptance:
            return None
        if not any(np.isfinite(list(self.ratio.values()))):
            raise ValueError('nan ratio')
        if max(self.curr_acceptance) < AcceptanceParams.acc:  # sometimes want to filter too low acc
            self.curr_acceptance = 0

        if AcceptanceParams.use_random_u:
            accepted_count = np.count_nonzero(
                np.array(self.curr_acceptance) > self.u_random_curr_iter)
        else:
            accepted_count = np.count_nonzero(
                np.array(self.curr_acceptance) > AcceptanceParams.acc)
        return accepted_count

    @staticmethod
    def get_posterior(liks_obj, priors_obj):
        """ Compute posterior for some partition"""
        liks_obj.compute_likelihood()
        priors_obj.compute_priors()

    def get_acceptance(self, change_track):
        if not self.first_iteration_done:
            self.get_posterior(self.curr_liks_obj, self.curr_priors_obj)
            self.first_iteration_done = True  # should compute only first time
        self.get_posterior(self.proposed_liks_obj, self.proposed_priors_obj)
        self.ratio = {}

        all_keys_list = LogicParams.parts_.keys_to_use_for_estimation_pairs
        for pair in all_keys_list:
            ratio = self.compute_ratio(pair, change_track)
            self.ratio[pair] = ratio
            self.analyse_ratio(pair)

        return self.acceptance

    def analyse_ratio(self, pair):
        if np.isfinite(self.ratio[pair]):
            if self.ratio[pair] == 1:
                self.acceptance[pair] = 0
            else:
                self.acceptance[pair] = min(1, self.ratio[pair])
        else:
            self.acceptance[pair] = 0

    def compute_ratio(self, pair, change_track):

        priors_curr_diff_new = list((Counter(self.curr_priors_obj.priors_numbers) - (
            Counter(self.proposed_priors_obj.priors_numbers))).elements())
        priors_new_diff_curr = list((Counter(self.proposed_priors_obj.priors_numbers) - Counter(
            self.curr_priors_obj.priors_numbers)).elements())
        priors_d = (np.prod(np.array(priors_new_diff_curr)) / np.prod(
            np.array(priors_curr_diff_new)))
        proposed_likelihoods_for_consideration = self.proposed_liks_obj.sort_by_pairs()
        current_likelihoods_for_consideration = \
            self.curr_liks_obj.sort_by_pairs(particular_ids=change_track['current'])
        try:
            proposed_likelihoods_for_consideration_curr_pair = proposed_likelihoods_for_consideration[pair]
            current_likelihoods_for_consideration_curr_pair = current_likelihoods_for_consideration[pair]
        except KeyError:
            logging.info('no states for {} pair'.format(pair))
            return 0

        lkls_new_diff_curr = list((Counter(proposed_likelihoods_for_consideration_curr_pair) - (
            Counter(
                current_likelihoods_for_consideration_curr_pair))).elements())  # for precision and performance, consider only difference
        lkls_curr_diff_new = list((Counter(current_likelihoods_for_consideration_curr_pair) - Counter(
            proposed_likelihoods_for_consideration_curr_pair)).elements())
        lkls_d = (
                util.count_log_lkl_by_list(lkls_new_diff_curr) /
                util.count_log_lkl_by_list(lkls_curr_diff_new))

        ratio = priors_d * lkls_d
        return ratio

    def choose_likelihoods_of_difference(self, pair, change_track):
        likelihoods_we_need = []
        for id in change_track['current']:
            likelihoods_we_need.append(
                self.curr_liks_obj.likelihoods_numbers[id][pair])
        likelihoods_we_need = [
            i for subl in likelihoods_we_need for i in subl]
        return likelihoods_we_need

    def accepted_(self, change_track=None):
        self.accepted = True
        self.curr_priors_obj = self.proposed_priors_obj

        for id_ in change_track['current']:
            self.curr_liks_obj.delete_likelihoods_by_id(id_)
        self.curr_liks_obj.add_likelihoods(self.proposed_liks_obj.likelihoods_numbers)


class Likelihoods:
    def __init__(self, states_likelihoods_need):
        self.filter = Filter()
        self.__states = states_likelihoods_need
        self.__likelihoods = {}
        self.likelihood = {}

    @property
    def likelihoods_numbers(self):
        return self.__likelihoods

    def delete_likelihoods_by_id(self, id):
        try:
            del self.__likelihoods[id]
        except:
            raise ValueError('cannot delete such id')

    def add_likelihoods(self, new_liks_id):
        self.__likelihoods.update(new_liks_id)

    def compute_likelihood(self):
        for track_index, track_state in self.__states.items():
            assert track_index not in MetaProcessingParams.false_indexes
            self.__likelihoods[track_index] = {}
            for pair_name, pair_state in track_state.items():
                if pair_name not in LogicParams.parts_.keys_to_use_for_estimation_pairs:
                    continue
                final_states = np.stack(pair_state, axis=1)
                assert len(final_states) > 2  # for kalman
                self.find_single_likelihood(
                    final_states, pair_name, track_index)
            if 'person' in track_state:
                self.similarities_pose, pose_2 = posu.compute_pose_similarity_score(
                    track_state['person'])
                pose_states = np.stack(
                    [self.similarities_pose, pose_2], axis=1)
                self.find_single_likelihood(
                    pose_states, 'person', track_index)

    def find_single_likelihood(
            self, final_states, pair_name, track_index):
        mu, cov, likelihoods = self.filter.get_likelihoods_with_kalman_filter(
            final_states)
        if likelihoods:
            self.__likelihoods[
                track_index][
                pair_name] = likelihoods

    def sort_by_pairs(self, particular_ids=None):
        new_likelihoods = {}
        for track_id, likelihoods_pairs in self.likelihoods_numbers.items():
            if particular_ids:
                if track_id not in particular_ids:
                    continue
            for pair_name, curr_pair_likelihoods in likelihoods_pairs.items():
                if pair_name not in new_likelihoods:
                    new_likelihoods[pair_name] = []
                new_likelihoods[pair_name].append(curr_pair_likelihoods)
        for pair, pair_liks in new_likelihoods.items():
            new_likelihoods[pair] = [
                i for subl in pair_liks for i in subl]
        return new_likelihoods


class Priors:
    def __init__(self, arr):
        self.arr = arr
        self._priors = None
        self.e_t_factrs = None
        self.a_t = None
        self.z_t = None
        self.c_t = None
        self.d_t = None
        self.f_t = None
        self.g_t = None
        self.indexes_for_every_frame = None
        self.e_t_1 = None
        self.tracks_numbers_at_curr_frame = None
        self.tracks_numbers_at_prev_frame = None
        self.det_falses = None
        self.__priors = None
        self.process_meta()

    @staticmethod
    def compute_single_prior(e_t, z_t, c_t, d_t, g_t, a_t, f_t):
        curr_prior = e_t * (AcceptanceParams.p_z ** z_t) * ((1 - AcceptanceParams.p_z) ** c_t) * \
                     (AcceptanceParams.p_d ** d_t) * ((1 - AcceptanceParams.p_d) ** g_t) * (
                             AcceptanceParams.lambda_b ** a_t) * (AcceptanceParams.lambda_f ** f_t)
        return curr_prior

    def process_meta(self):
        indexes_for_every_frame_ = np.split(self.arr[:, 1], np.cumsum(
            np.unique(self.arr[:, 0], return_counts=True)[1])[:-1])
        indexes_for_every_frame = [list(index)
                                   for index in indexes_for_every_frame_]
        self.indexes_for_every_frame = list(
            map(pdu.remove_str_from_indexes, indexes_for_every_frame))
        len_indexes_for_every_frame = list(
            map(pdu.get_len_single, indexes_for_every_frame))
        self.det_falses = list(
            map(pdu.get_false_inds_and_detections, indexes_for_every_frame[1:]))
        self.e_t_1 = len_indexes_for_every_frame[:-1]
        self.tracks_numbers_at_curr_frame = indexes_for_every_frame[1:]
        self.tracks_numbers_at_prev_frame = indexes_for_every_frame[:-1]

    def get_characteristics_priors(self):
        """ Compute characteristics according to the article."""

        self.e_t_factrs = list(
            map(pdu.get_len_single_fact, self.indexes_for_every_frame[1:]))
        self.a_t = list(map(pdu.diff_consecutive_frames,
                            self.tracks_numbers_at_curr_frame,
                            self.tracks_numbers_at_prev_frame))
        self.z_t = list(map(pdu.diff_consecutive_frames, self.tracks_numbers_at_prev_frame,
                            self.tracks_numbers_at_curr_frame))
        self.c_t = [a - b for a, b in zip(self.e_t_1, self.z_t)]
        self.d_t = list(np.array(self.det_falses)[:, 0:1].T[0])
        self.f_t = list(np.array(self.det_falses)[:, 1:2].T[0])
        self.g_t = [
            a - b + c - d for a,
                              b,
                              c,
                              d in zip(
                self.e_t_1,
                self.z_t,
                self.a_t,
                self.d_t)]

    def compute_priors(self):
        """ Compute priors."""
        self.get_characteristics_priors()
        curr_priors = list(map(self.compute_single_prior,
                               self.e_t_factrs, self.z_t, self.c_t, self.d_t, self.g_t, self.a_t, self.f_t))
        self.__priors = curr_priors

    @property
    def priors_numbers(self):
        return self.__priors


class Filter:
    def __init__(self):
        dt = 1
        self.matrix_a = np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

        self.matrix_g = np.array([[(dt ** 2) / 2, 0],
                                  [0, (dt ** 2) / 2],
                                  [dt, 0],
                                  [0, dt]])

        self.matrix_c = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.r = np.zeros((2, 2), int)
        np.fill_diagonal(self.r, KalmanParams.r)
        self.q = np.zeros((4, 4), int)
        np.fill_diagonal(self.q, KalmanParams.q)
        self.filter = self.initialise_filter()
        logging.debug('filter initialised')

    def initialise_filter(self):
        filter_ = KalmanFilter(dim_x=4,
                               dim_z=2)  # need to instantiate every time to reset all fields
        filter_.F = self.matrix_a
        filter_.H = self.matrix_c
        filter_.B = self.matrix_g

        if KalmanParams.use_noise_in_kalman:
            u = np.random.normal(loc=0, scale=KalmanParams.var_kalman, size=2)
            filter_.u = u
        # u = Q_discrete_white_noise(dim=2, var=1)

        filter_.Q = self.q
        filter_.R = self.r
        return filter_

    def get_likelihoods_with_kalman_filter(self, states_info):
        self.initialise_filter()
        initial_state = [states_info[1][0], states_info[1][1], states_info[1][0] - states_info[0][0],
                         states_info[1][1] - states_info[0][1]]
        assert not np.all(np.isnan(initial_state))
        assert states_info[1][0] != 0 and states_info[1][1] != 0

        self.filter.x = np.array([initial_state[0], initial_state[1], initial_state[2],
                                  initial_state[3]]).T
        states_info = states_info[2:]
        mu = []
        cov = []

        likelihoods, xs, xu, means, covariances, means_p, covariances_p = self.filter.batch_filter(np.array(
            states_info))
        return mu, cov, likelihoods


class ExtendedPartition:
    def __init__(self, partition, grouped_ids, states):
        self.partition = partition
        self.grouped_ids = grouped_ids
        self.states = states

    class Memento(object):
        def __init__(self, mstate):
            self.mstate = mstate

        def rollback_state(self):
            return self.mstate

    def set_state(self, state):
        self.__current_state = state

    @property
    def curr_st(self):
        return self.__current_state

    def save_state(self):
        return self.Memento(copy.deepcopy(self))

    def rollback_state(self, memento):
        self = memento.rollback_state()
        print('rollback to state {} '.format(self.curr_st))
