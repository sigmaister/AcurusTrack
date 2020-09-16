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

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def compute_pose_similarity_score(persons):
    sims_scores = []
    sims_scores2 = []
    prev_person = persons[0]
    for index, single_person in enumerate(persons):
        if index == 0:
            continue
        if not isinstance(single_person, list):
            continue
        eu, cosin = compare_two_poses(persons[index], prev_person)
        sims_scores.append(eu)
        sims_scores2.append(cosin)
        prev_person = single_person
    return sims_scores, sims_scores2


def compare_two_poses(pose1, pose2, confidence_score=0.1):
    pose_1_filtered = []
    pose_2_filtered = []
    for p_1, p_2 in zip(pose1, pose2):
        if sum(p_1) > 0 and sum(
                p_2) > 0 and p_2[2] > confidence_score and p_1[2] > confidence_score:
            pose_1_filtered.append(p_1[0])
            pose_1_filtered.append(p_1[1])
            pose_2_filtered.append(p_2[0])
            pose_2_filtered.append(p_2[1])
    cosine_sim_score = cos_sim(pose_1_filtered, pose_2_filtered)
    return np.sqrt(2 * (1 - cosine_sim_score)), cosine_sim_score
