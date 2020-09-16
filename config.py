import matplotlib.colors as mcolors
import numpy as np

from dataclasses import dataclass


def create_colors():
    """
    remove pale colors from palette
    :return:cleaned colors and their names
    """
    colors = mcolors.CSS4_COLORS
    color_names_original = list(colors)
    color_names_to_delete = ['white', 'antiquewhite', 'floralwhite', 'whitesmoke', 'navajowhite', 'ghostwhite',
                             'snow', 'oldlace', 'ivory', 'seashell', 'mistyrose', 'lavender', 'lavenderblush',
                             'azure', 'lightyellow', 'beige', 'lightgoldenrodyellow', 'honeydew', 'cornsilk',
                             'lightgrey', 'darkgrey', 'mintcream', 'aliceblue', 'papayawhip', 'moccasin',
                             'lightslategray',
                             'lightgray', 'grey']

    color_names = list(set(color_names_original) - set(color_names_to_delete))
    rgb_colors = [(mcolors.to_rgb(color)) for color in color_names]
    rgb_colors_new = []
    for y in rgb_colors:
        rgb_colors_new.append(tuple([int(i * 255) for i in y]))
    return rgb_colors_new, color_names


@dataclass
class SystemParams:
    Pattern = "([^(]*)"
    seed = 0
    use_multiprocessing = True  # split video into chunks and process them in parallel


@dataclass
class FilesProcessingParams:
    key_1_for_choose_from_dir = "LAST_TRUE"
    key_2_for_choose_from_dir = "BEST_HAT"
    final_meta_name = 'meta_after_final_merge'  # name for meta got after final merge


@dataclass
class PoseParams:
    keys_to_use_for_estimation_pairs = [('MidHip_x', 'MidHip_y'), ('BigToes_x', 'BigToes_y'),
                                        ('Neck_x',
                                         'Neck_y')]  # pairs used for pose processing and acceptance calculation
    keys_to_use_for_estimation = ['MidHip_x', 'MidHip_y', 'BigToes_x', 'BigToes_y',
                                  'Neck_x', 'Neck_y']
    keys_to_use_for_fcs = keys_to_use_for_estimation_pairs


@dataclass
class FaceParams:
    keys_to_use_for_estimation_pairs = [('center_x', 'center_y')]
    keys_to_use_for_estimation = ['center_x', 'center_y']
    keys_to_use_for_fcs = [('center_x', 'center_y'), ('x1', 'y1'), ('x2', 'y2')]
    keys_pairs = ["x1", "x2", "y1", "y2"]


@dataclass
class FaceInitialisationParams:
    iou_threshold = 0.4  # for initialisation
    ids_counter = 0  # id to start numeration from


@dataclass
class PoseInitialisationParams:
    ids_counter = 0  # id to start numeration from
    error_threshold = 40  # threshold for pose initialisation
    confidence_score = 0.1  # filter pose meta
    curr_indexes_to_consider = {'MidHip': 8, 'Neck': 1, 'BigToes': None, 'Nose': 0}  # indexes to consider in pose meta
    len_pose_to_compare = 2  # minimal amount of estimated pose parts to consider this person


@dataclass
class LogicParams:
    face_or_pose_meta = 'face'  # meta processing nature
    parts_ = PoseParams if face_or_pose_meta == 'pose' else FaceParams
    init_params = PoseInitialisationParams if face_or_pose_meta == 'pose' else FaceInitialisationParams
    use_final_merge = False  # use final merge or finish at concatenated


@dataclass
class DrawingParams:
    # draw or not "false" tracks , considered as trash
    draw_false_tracks = False
    # draw or not every iteration at final merge
    draw_every_iteration = False
    # draw or not accepted moves
    draw_accepted = False
    rgb_colors_new, colors_names = create_colors()
    # -1 for filling rectangle
    thickness = -1
    # overlay weights
    alpha = 0.6
    alpha_text = 0.55
    alpha_text_id = 0.3
    alpha_id = 0.5
    # draw full pose or just some parts
    full_pose = True
    # confidence threshold to draw pose
    conf_threshold = 0.2
    # pose pairs to be used when drawing
    pose_pair_connections = [(0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (9, 10), (10, 11),
                             (8, 9), (8, 12), (12, 13), (13, 14), (15, 16), (17, 18), (19, 20), (20, 21), (22, 23),
                             (23, 24)]
    # draw or not file name at the top right angle
    draw_file_name = False
    draw_ids_counter = False
    draw_logo = True


@dataclass
class KalmanParams:
    use_noise_in_kalman = True
    var_kalman = 0.2
    q = 1
    r = 1


@dataclass
class AcceptanceParams:
    p_z = 0.2  # probability of object disappearing
    p_d = 0.9  # detection probability
    lambda_b = 0.2  # birth rate of new objects per unit time, per unit volume
    lambda_f = 0.001  # the false alarm rate per unit time, per unit volume
    acc = 0.1
    number_of_acc_for_acc = 1  # how much parts should satisfy by acceptance condition to be considered as accepted move
    use_random_u = False


@dataclass
class MetaProcessingParams:
    len_to_make_trash_index = 2  # throw out tracks with length less or equal this parameter
    false_indexes = [-1, -10, -100, -200,
                     np.nan]
    max_tracks_number_at_window = 30
    renumbering = True
    overlap = 10
    fixed_coordinate_system = True
