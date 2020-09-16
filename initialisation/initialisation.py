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

from abc import ABC, abstractmethod

from config import LogicParams


class MetaInitialisation(ABC):
    @abstractmethod
    def __init__(self, meta):
        self.meta = meta
        self.ids_counter = LogicParams.init_params.ids_counter
        self.first_frame_passed = False
        self.prev_frame_persons_list = None
        self.persons_info = []
        self.prev_frame_person = None
        self.current_frame_person = None
        self.curr_persons_errors = {}
        self.curr_person_distances = {}
        self.chosen_indexes = []

    def initialisation(self):

        new_meta = {}
        assert self.meta != []

        for frame_no, frame_meta in self.meta.items():
            frame_no_int = int(frame_no)
            self.initialisation_internal(frame_meta)
            self.prev_frame_persons_list = self.persons_info
            new_meta[frame_no_int] = self.persons_info

        return new_meta

    @abstractmethod
    def analyse_distances(self, person):
        raise NotImplementedError("Must override analyse_distances")

    @abstractmethod
    def looking_for_candidate_person(self):
        raise NotImplementedError("Must override looking_for_candidate_person")

    def analyse_two_consecutive_frames(self, frame_info):
        assert self.prev_frame_persons_list is not None
        self.chosen_indexes = []
        for person in frame_info:
            self.current_frame_person = person
            self.curr_persons_errors = {}
            self.curr_person_distances = {}
            for prev_person in self.prev_frame_persons_list:
                self.prev_frame_person = prev_person
                self.looking_for_candidate_person()
            self.analyse_distances(person)

    @abstractmethod
    def set_info(self, person_full, curr_index):
        raise NotImplementedError("Must override get_info")

    def initialisation_internal(self, frame_info):
        self.persons_info = []
        if not self.first_frame_passed:
            # for the first frame assign as many ids as there are
            for index, person in enumerate(frame_info):
                new_info = self.set_info(person, index)
                self.persons_info.append(new_info)
                self.ids_counter += 1
            self.first_frame_passed = True
        else:
            self.analyse_two_consecutive_frames(frame_info)






