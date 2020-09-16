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


def euclidean_norm(y, y_t_d):
    new_x = y['x1'] - y_t_d['x1']
    new_y = y['y1'] - y_t_d['y1']
    return (new_x ** 2 + new_y ** 2) ** 0.5


def euclidean_norm_pose(y, y_t_d):
    new_1 = y[0] - y_t_d[0]
    new_2 = y[1] - y_t_d[1]
    return (new_1 ** 2 + new_2 ** 2) ** 0.5
