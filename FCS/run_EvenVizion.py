import argparse
import os

import cv2
import json

from evenvizion.processing import video_processing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments without using Sacred library')
    parser.add_argument('--path_to_video',
                        default="/media/kulichd/data/Data/EvenVizion/package/camera-motion-detection/EvenVizion/examples/test_video/test_video.mp4")
    args = parser.parse_args()
    args.save_folder = main_dir = os.path.dirname(args.path_to_video)
    path_to_homography_dict = "{}/dict_with_homography_matrix.json".format(args.save_folder)
    print(path_to_homography_dict)
    cap = cv2.VideoCapture(args.path_to_video)
    homography_dict = video_processing.get_homography_dict(cap)
    with open(path_to_homography_dict, "w") as json_:
        json.dump(homography_dict, json_)


