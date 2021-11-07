# 이미지 왜곡 보정

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path as path
import pickle


def lazy_calibration(func):
    """
    Decorator for calibration function to avoid re-computing calibration every time.
    매번 교정을 재계산하지 않도록 교정 기능을 위한 데코레이터.
    """
    calibration_cache = 'camera_cal/calibration_data.pickle'

    def wrapper(*args, **kwargs):
        if path.exists(calibration_cache): # 파일 있는지 체크
            print('Loading cached camera calibration...', end=' ')
            with open(calibration_cache, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
                print(calibration)
                print(type(calibration)) # 투플
        else:
            print('Computing camera calibration...', end=' ')
            calibration = func(*args, **kwargs)
            with open(calibration_cache, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)
        print('Done.')
        return calibration

    return wrapper


@lazy_calibration
def calibrate_camera(calib_images_dir, verbose=False):
    """
    Calibrate the camera given a directory containing calibration chessboards.
    보정 체스판이 들어 있는 디렉터리를 사용하여 카메라를 보정합니다.
    :param calib_images_dir: directory containing chessboard frames 체스판 프레임이 들어 있는 디렉터리
    :param verbose: if True, draw and show chessboard corners True이면 체스판 모서리를 그리고 표시합니다.
    :return: calibration parameters
    """
    
    # assert [조건] , 오류메시지
    assert path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)

    # np.zeros()는 주어진 shape나 type에 맞춰 0으로 채워진 array 반환
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # np.mgrid [0 : 8,0 : 6]은 다음 mgrid를 생성합니다.
    # array([[[0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1],
    #         [2, 2, 2, 2, 2, 2],
    #         [3, 3, 3, 3, 3, 3],
    #         [4, 4, 4, 4, 4, 4],
    #         [5, 5, 5, 5, 5, 5],
    #         [6, 6, 6, 6, 6, 6],
    #         [7, 7, 7, 7, 7, 7]],
    #
    #        [[0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 1, 2, 3, 4, 5]]])

    # Arrays to store object points and image points from all the images. 모든 이미지의 객체 지점 및 이미지 지점을 저장하는 배열입니다.
    objpoints = []  # 3d points in real world space 실제 공간의 3D 점
    imgpoints = []  # 2d points in image plane 영상 평면의 2d 점

    # Make a list of calibration images 보정 이미지 목록 만들기 (glob이 파일리스트 뽑는거)
    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners 이미지 목록을 단계별로 살펴보고 체스판 모서리를 검색
    for filename in images:

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None) # (이미지, 패턴사이즈,

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, pattern_found)
                cv2.imshow('img',img)
                cv2.waitKey(500)

    if verbose:
        cv2.destroyAllWindows()

    # calibrateCamera : camera 행렬, 왜곡계수, 회전/변환 벡터들 리턴
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort(frame, mtx, dist, verbose=False):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    카메라 매트릭스와 왜곡 계수가 지정된 프레임을 분리합니다.
    :param frame: input frame 입력 프레임
    :param mtx: camera matrix 카메라 행렬
    :param dist: distortion coefficients 왜곡 계수
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx) # 왜곡 보정

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_undistorted


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    img = cv2.imread('curvature_test_img/16_14-58-21.png')

    img_undistorted = undistort(img, mtx, dist, True)

    cv2.imwrite('curvature_test_img/16_14-58-21_before.png', img)
    cv2.imwrite('curvature_test_img/16_14-58-21_after.png', img_undistorted)