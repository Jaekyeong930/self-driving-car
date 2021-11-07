# 원근변환
# 차선을 인식할 범위를 지정

import numpy as np
import cv2
import glob # glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize


def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame 입력하는 컬러 이미지
    :param verbose: if True, show the transformation result 참이면 변환된 결과값(이미지)을 보여준다
    :return: warped image, and both forward and backward transformation matrices
    리턴값으로 휘어진 이미지, 순방향 및 역방향 변환 행렬(원근 보정된) 모두를 반환함
    """

    # 좌표값 범위 지정

    h, w = img.shape[:2] # 이미지의 높이와 너비

    # 원래 코드에 있던 소스
    # src = np.float32([[w, h-10],    # br 우측밑
    #                       [0, h-10],    # bl 좌측밑
    #                       [546, 460],   # tl 좌측위
    #                       [732, 460]])

    # src = source 출발지 / dst = destination 목적지
    src = np.float32([[w, 400],    # br 우측밑
                      [0, 400],    # bl 좌측밑
                      [300, 200],   # tl 좌측위
                      [1000, 200]])  # tr 우측위
    # 어디까지 늘릴건지?
    dst = np.float32([[w, h],       # br 우측밑
                      [0, h],       # bl 좌측밑
                      [0, 0],       # tl 좌측위
                      [w, 0]])      # tr 우측위

    M = cv2.getPerspectiveTransform(src, dst) # 투시 변환 행렬을 구함 (뒤틀린 행렬을 보정하기 위한 첫단계)
    Minv = cv2.getPerspectiveTransform(dst, src) # 다시 반대로 보정

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR) # 투시 변환 결과값 출력
    # (입력영상, 투시 변환 행렬, 결과 영상의 크기, 보간법)

    # cv2.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst
    #
    # • src: 입력 영상
    # • M: 3x3 투시 변환 행렬. 실수형.
    # • dsize: 결과 영상 크기. (w, h) 튜플. (0, 0)이면 src와 같은 크기로 설정
    # • dst: 출력 영상
    # • flags: 보간법. 기본값은 cv2.INTER_LINEAR
    # • borderMode: 가장자리 픽셀 확장 방식. 기본값은 cv2.BORDER_CONSTANT
    # • borderValue: cv2.BORDER_CONSTANT일 때 사용할 상수 값. 기본값은 0.

    if verbose:
        # 프로세싱 input값과 output값을 어떻게 배치할 것인지 정함 (1x2행렬)
        f, axarray = plt.subplots(1, 2) #  행, 열의 개수를 인자로 받아서 figure(수치) 및 axes(축) 객체를 포함하는 튜플을 반환
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray') # cmap은 컬러맵 변경
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # show result on test images
    for test_img in glob.glob('16cap/*.png'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary = binarize(img_undistorted, verbose=False)

        img_birdeye, M, Minv = birdeye(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), verbose=True)


