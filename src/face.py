import cv2
import numpy as np
from skimage import transform as trans



FACE_SIZE = (96, 112)

def align(method, img, landmarks):
    """ Aligns a face image into a canonical position
    Parameters
    ----------
    method: str,
        Supported methods = ["skimage", "opencv_affine", "opencv_affine_partial"]
    img: numpy array of floats
        Either RGB or greyscale image. Should be shape (height, width, 1) if greyscale and (height, width,  3) if RGB
    landmarks: a list of tuples.
        Should contain the coordinates of the 5 facial landmarks in the following order:
            - left eye
            - right eye
            - nose
            - left mouth corner
            - right mouth corner
        Preferably use MTCNN to extract these.
    Returns
    -------
    The aligned image with size (height = 112, width = 96)
    """

    supported_methods = ["skimage", "opencv_affine", "opencv_affine_partial"]
    assert method in supported_methods, "method not supported: {0}".format(method)

    if method == "skimage":
        return _align_skimage(img, landmarks)
    if method == "opencv_affine":
        return _align_opencv_affine(img, landmarks)
    if method == "opencv_affine_partial":
        return _align_opencv_affine_partial(img, landmarks)


canonical_positions = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)


def _align_skimage(img, landmarks):
    """
    Parameters
    ----------
    img: numpy array of floats
        Either RGB or greyscale image. Should be shape (height, width, 1) if greyscale and (height, width,  3) if RGB
    landmarks: a list of tuples.
        Should contain the coordinates of the 5 facial landmarks in the following order:
            - left eye
            - right eye
            - nose
            - left mouth corner
            - right mouth corner
        Preferably use MTCNN to extract these.
    Returns
    -------
    The aligned image with size (height = 112, width = 96)
    """

    # validate input
    assert img.shape[2] in [1, 3], "Image shape error"
    assert len(landmarks) == 5, "landmarks shape error"

    # Some HACK seen in https://github.com/Joker316701882/Additive-Margin-Softmax/blob/master/align/face_preprocess.py
    # Don't know the reason.
    dst = canonical_positions
    landmarks = np.array(landmarks).astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, FACE_SIZE, borderValue=0.0)
    return warped


def _align_opencv_affine(img, landmarks):
    """
    Parameters
    ----------
    img: numpy array of floats
        Either RGB or greyscale image. Should be shape (height, width, 1) if greyscale and (height, width,  3) if RGB
    landmarks: a list of tuples.
        Should contain the coordinates of the 5 facial landmarks in the following order:
            - left eye
            - right eye
            - nose
            - left mouth corner
            - right mouth corner
        Preferably use MTCNN to extract these.
    Returns
    -------
    The aligned image with size (height = 112, width = 96)
    """

    # validate input
    assert img.shape[2] in [1, 3], "Image shape error"
    assert len(landmarks) == 5, "landmarks shape error"

    # Some HACK seen in https://github.com/Joker316701882/Additive-Margin-Softmax/blob/master/align/face_preprocess.py
    # Don't know the reason.
    dst = canonical_positions
    landmarks = np.array(landmarks).astype(np.float32)

    transmat = cv2.estimateAffine2D(landmarks, dst)[0]
    warped = cv2.warpAffine(img, transmat, FACE_SIZE, borderValue=0.0)

    return warped


def _align_opencv_affine_partial(img, landmarks):
    """
    Parameters
    ----------
    img: numpy array of floats
        Either RGB or greyscale image. Should be shape (height, width, 1) if greyscale and (height, width,  3) if RGB
    landmarks: a list of tuples.
        Should contain the coordinates of the 5 facial landmarks in the following order:
            - left eye
            - right eye
            - nose
            - left mouth corner
            - right mouth corner
        Preferably use MTCNN to extract these.
    Return
    -------
    The aligned image with size (height = 112, width = 96)
    """

    # validate input
    assert img.shape[2] in [1, 3], "Image shape error"
    assert len(landmarks) == 5, "landmarks shape error"

    # Some HACK seen in https://github.com/Joker316701882/Additive-Margin-Softmax/blob/master/align/face_preprocess.py
    # Don't know the reason.
    dst = canonical_positions
    landmarks = np.array(landmarks).astype(np.float32)

    transmat = cv2.estimateAffinePartial2D(landmarks, dst)[0]
    warped = cv2.warpAffine(img, transmat, FACE_SIZE, borderValue=0.0)

    return warped


