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

import math

import numpy as np
import pandas as pd

from config import MetaProcessingParams, LogicParams

"""This file contains common utils functions for processing pandas dataframes during mcmcda algorithm"""


def print_pandas(dataframe):
    """ Print pandas dataframe in "full" mode """

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataframe)


def starts(dataframe):
    """ Find start frame numbers of the tracks"""
    return pd.DataFrame(
        dataframe[dataframe['frame_no'].diff() != 1]['frame_no'])


def ends(dataframe):
    """ Find end frame numbers of the tracks"""
    return pd.DataFrame(
        dataframe[dataframe['frame_no'].diff(periods=-1) != -1]['frame_no'])


def change_index_in_df(dataframe, old, new, start=None, end=None):
    """ Replace indexes in dataframes in specified range """
    if start and end:
        dataframe.loc[(dataframe.id == old) & (dataframe.frame_no >=
                                               start) & (dataframe.frame_no <= end), 'id'] = new
    elif start:
        dataframe.loc[(dataframe.id == old) & (dataframe.frame_no >=
                                               start), 'id'] = new
    elif end:
        dataframe.loc[(dataframe.id == old) & (dataframe.frame_no <= end), 'id'] = new
    else:
        dataframe.loc[dataframe.id == old, 'id'] = new

    return dataframe


def update_candidate_pairs(pair, pairs_cleaned, new_id):
    """ Update candidate pairs, considering moves been done"""
    new_pairs = []
    pairs_contained_first_id = []
    pairs_contained_second_id = []
    for pair_curr in pairs_cleaned:
        if pair == pair_curr:
            continue
        if pair[0] in pair_curr:
            pairs_contained_first_id.append(
                [i for i in pair_curr if i != pair[0]][0])
        elif pair[1] in pair_curr:
            pairs_contained_second_id.append(
                [i for i in pair_curr if i != pair[1]][0])
        else:
            new_pairs.append(pair_curr)
    intersect = set(pairs_contained_first_id).intersection(
        set(pairs_contained_second_id))
    intersected_pairs = [(new_id, i) for i in intersect]
    final_pairs = list(set(new_pairs).union(set(intersected_pairs)))
    return final_pairs


def check_intersection(pair):
    """ Check if tracks from pairs do intersect"""
    return set(pair[0]).intersection(pair[1]) != set()


def get_current_meta_indexes(pandas_dataframe):
    """ Return unique detection indexes"""
    unique_not_false_inds = list(
        set(list(pandas_dataframe.id.unique())) - set(MetaProcessingParams.false_indexes))
    unique_not_false_inds = [
        index for index in unique_not_false_inds if not np.isnan(index)]
    return unique_not_false_inds


def from_dataframe_to_dict(dataframe):
    """ Return dictionary with meta from pandas dataframe."""
    check_for_dublicates(dataframe)
    df_copy = dataframe.copy(deep=True)
    df_copy.rename(columns={'id': 'index'}, inplace=True)
    df_copy = df_copy.sort_values(by=['frame_no'])
    meta_in_dict_format = {frame_no: list(df_copy.xs(frame_no).to_dict(
        'index').values()) for frame_no in df_copy.frame_no}
    return meta_in_dict_format


def read_multiindex_pd(path_to_pandas_dataframe):
    """Read dataframe from csv file."""
    dataframe = pd.read_csv(path_to_pandas_dataframe, index_col=[0, 1])
    return dataframe


def dataframe_from_dict(our_meta):
    """ Make pandas dataframe from the dict with meta

    """
    new_meta = {}
    for frame_no, frame_info in our_meta.items():
        new_meta[frame_no] = {}
        for index, elem in enumerate(frame_info):
            new_meta[frame_no][index] = elem
    dataframe = pd.DataFrame.from_dict({(i, j): new_meta[i][j]
                                        for i in new_meta
                                        for j in new_meta[i]},
                                       orient='index')
    dataframe['frame_no'] = dataframe.index.get_level_values(0)
    dataframe.rename(columns={'index': 'id'}, inplace=True)
    return dataframe


def check_for_dublicates(dataframe):
    """ Check if there are the same ids on the same frame - we suppose that this is impossible
    and should not be processed

    """
    df1 = dataframe[~dataframe['id'].isin(MetaProcessingParams.false_indexes)].groupby(
        [dataframe.id])  # allow false indexes to repeat on the same frame
    boolean_result = df1['frame_no'].diff() == 0
    if boolean_result.any():
        print_pandas(boolean_result)
        raise ValueError('there are repetitions!!!')


def get_len_single(indexes):
    """ Return number of indexes on the single frame"""
    if indexes == ['']:
        return 0
    return len(indexes)


def get_len_single_fact(indexes):
    """ Return value needed for priors calculation"""
    assert indexes != ['']
    return 1 / math.factorial(len(indexes))


def get_false_inds_and_detections(indexes):
    """ Return number of detections, quantity of false indexes"""
    number_of_detections = len(list(set(indexes) - set(MetaProcessingParams.false_indexes)))
    false_indexes = len(indexes) - number_of_detections
    return [number_of_detections, false_indexes]


def diff_consecutive_frames(tracks_inds_prev_frame, tracks_inds_next_frame):
    """ Return difference between consecutive frames indexes"""
    return len(list(set(tracks_inds_prev_frame) - set(tracks_inds_next_frame)))


def get_frames_indexes(dataframe):
    """ Return list of indexes"""
    return dataframe['id'].values.tolist()


def remove_str_from_indexes(indexes):
    """ Remove str from indexes"""
    if indexes == ['']:
        return []
    return indexes


def get_tracks(dataframe_meta, names):
    """

    :param dataframe_meta:  dataframe with detections in specified format
    :param names: list of keys to consider
    :return: coordinates of specified keys, which can be used as observations for Kalman filter,
    missing values are filled with np.nan

    """

    range_ = np.arange(
        dataframe_meta.index[0][0], dataframe_meta.index[-1][0] + 1)
    dropped = dataframe_meta.reset_index(level=1, drop=True)
    fixed_dataframe = dropped.reindex(range_)
    all_body_parts_tracks = {}
    for body_part_pair in names:
        curr_pair_states = body_parts_pair_processing(
            fixed_dataframe, body_part_pair)
        if curr_pair_states:
            all_body_parts_tracks[tuple(body_part_pair)] = curr_pair_states
    return all_body_parts_tracks


def body_parts_pair_processing(
        fixed_dataframe, body_part_pair, minimal_len_to_consider=3):
    """ Return some body parts coordinates."""
    curr_pair_states = []
    for key in body_part_pair:
        curr_body_part_track = fixed_dataframe[key].values.tolist()

        no_zeros_meta = clean_pose(curr_body_part_track)
        if no_zeros_meta is None:
            break
        no_zeros_processed_meta = clean_nans_in_pose_meta(no_zeros_meta)
        if len(no_zeros_processed_meta) >= minimal_len_to_consider:
            curr_pair_states.append(no_zeros_processed_meta)
    return curr_pair_states


def clean_pose(body_part_track):
    """ Pose estimator return zero values for body parts it could not find,
     it should not be taken into consideration.

    """
    cleaned_track = []
    for index, elem in enumerate(body_part_track):
        if elem == 0:
            if index in [0, len(body_part_track) - 1]:
                continue
            if len(cleaned_track) >= 2:
                cleaned_track.append(np.nan)
        else:
            cleaned_track.append(elem)
    return cleaned_track if len(cleaned_track) > 1 else None


def clean_nans_in_pose_meta(body_part_track):
    """ Clean track, beginning from nan, as Kalman filter
    cannot be initialised with nans.

    """
    while np.isnan(body_part_track[0]) or np.isnan(
            body_part_track[1]):  # remove nans from the beginning for right kalman initialisation
        del body_part_track[0]
        if len(body_part_track) <= 1:
            break
    return body_part_track


def get_particular_states(dataframe, list_of_indexes):
    """ Return track coordinates for particular indexes."""
    states = {}
    for i in list_of_indexes:
        group1 = dataframe[dataframe.id == i].copy(deep=True)
        states[i] = get_tracks(
            group1, LogicParams.parts_.keys_to_use_for_estimation_pairs)
    return states


def get_frames_numbers(meta, global_dict):
    global_dict[meta['id'].values.tolist(
    )[0]] = meta['frame_no'].values.tolist()


def merge_two_consecutive_windows(
        prev_window_dataframe, next_window_dataframe, counter):
    """ Merge two overlapped windows. """
    intersected_rule = find_intersection_indexes(
        prev_window_dataframe, next_window_dataframe)
    replace_rule = clean_update_indexes_rule(intersected_rule)
    next_window_indexes = next_window_dataframe.id.unique()
    rest = list(set(next_window_indexes) - set(MetaProcessingParams.false_indexes) - set(
        list(replace_rule.keys())))
    replace = list(np.arange(counter, len(rest) + counter))
    counter += len(rest)
    replace_rule_no_duplicates = dict(zip(rest, replace))
    for key, value in replace_rule.items():  # remove set in values
        replace_rule[key] = list(value)[0]
    replace_rule.update(replace_rule_no_duplicates)
    next_df = next_window_dataframe.replace({'id': replace_rule})
    merged_final_df = pd.concat([prev_window_dataframe, next_df]).groupby(
        level=[0, 1]).last().drop_duplicates()
    return merged_final_df, counter


def find_intersection_indexes(prev_window_dataframe, next_window_dataframe):
    """ Find indexes which tracks match in overlapped region."""
    intersected_df = prev_window_dataframe.merge(next_window_dataframe, how='inner',
                                                 on=LogicParams.parts_.keys_to_use_for_estimation)
    replace_rule = pd.Series(intersected_df.id_x.values,
                             index=intersected_df.id_y)
    intersected_rule = replace_rule.groupby('id_y').apply(set).to_dict()
    return intersected_rule


def clean_update_indexes_rule(intersected_rule):
    """ Choose bijection. There are some cases when part of the tracks
    processed different in different windows because of the context.
    So in replace rule one indexes may correspond two or more indexes. We choose only one-to-one rule.

    """
    cleared_inds = {}
    for key in intersected_rule.keys():
        if len(intersected_rule[key]) == 1:  # consider one-to-one
            cleared_inds[key] = intersected_rule[key]
    inv_dict = {}
    for k, v in cleared_inds.items():
        new_key = list(v)[0]
        if new_key not in inv_dict:
            inv_dict[new_key] = []
        inv_dict[new_key].append(k)
    for k, v in inv_dict.items():
        if len(v) > 1:
            for ind in v:
                del cleared_inds[ind]
    return cleared_inds


def get_update_indexes_rule(dataframe):
    """ Return rule for updating indexes in consecutive manner"""
    indexes = get_current_meta_indexes(dataframe)
    assert not any(isinstance(x, str) for x in indexes)
    rest = list(set(indexes) - set(MetaProcessingParams.false_indexes))
    replace_indexes_new = list(np.arange(len(rest)))
    replace_rule_no_duplicates = dict(zip(rest, replace_indexes_new))
    return replace_rule_no_duplicates
