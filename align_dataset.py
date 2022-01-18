import os
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from mtcnn.mtcnn import MTCNN
from src import face


def parseargs():
    parser = ArgumentParser(description='Align dataset')

    parser.add_argument("--dataset", type=str,
                        help="String - path to the train set folder or file")
    parser.add_argument("--output", type=str,
                        help="String - path to the aligned dataset to be saved")

    return parser.parse_args()
    pass


def get_file_list(directory):
    """
    Parameters
    ----------
    img: str. Path to the directory to be searched for files.

    Returns all the files in a folder recursively
    """
    file_list = []
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            relative_root = root.replace(directory, "")
            file_list.append(os.path.join(relative_root, filename))
    return file_list


def main(dataset, output):
    """
    Main method to detect faces in the entire image dataset and save the aligned faces separately.
    """

    # Get file list
    file_list = get_file_list(dataset)
    # Define face detector
    face_detector = MTCNN()

    # Interate all files
    for f in tqdm(file_list):
        # Read image
        img = cv2.imread(os.path.join(dataset, f))
        # Svitch from BGR color space to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detec faces on the image
        detection = face_detector.detect_faces(img)
        # If no face is detected print the file name
        if len(detection) == 0:
            print("No face found in %s" % (f))
            continue
        # tahe the first detection
        keypoints = detection[0]["keypoints"]
        # Align the face
        landmarks = [keypoints["left_eye"],
                     keypoints["right_eye"],
                     keypoints["nose"],
                     keypoints["mouth_left"],
                     keypoints["mouth_right"]]

        face_image = face.align(method="opencv_affine",
                                img=img,
                                landmarks=landmarks)

        save_file = os.path.join(output, f)
        # make dir if does not exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        # save aligned face image to new loation
        cv2.imwrite(save_file, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    args = parseargs()
    main(**args.__dict__)
