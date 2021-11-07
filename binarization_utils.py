

# 이 코드를 활용해서 하수구 색을 검출하고 아예 다른색으로 지워버릴 수는 없을까 ?


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


# selected threshold to highlight yellow lines 노란색 선을 강조 표시하기 위해 선택된 임계값 (노란색의 범위)
yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])


def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):
    """
    Threshold a color frame in HSV space
    HSV 공간의 색상 프레임 임계값 지정
    """
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # 색상 속성을 BGR에서 HSV로 변경

    min_th_ok = np.all(HSV > min_values, axis=2) # min값보다 큰거 다 검출
    max_th_ok = np.all(HSV < max_values, axis=2) # max값보다 작은거 다 검출

    out = np.logical_and(min_th_ok, max_th_ok)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out


def thresh_frame_sobel(frame, kernel_size): # 엣지검출 / 커널 사이즈는 매트릭스의 크기
    """
    Apply Sobel edge detection to an input frame, then threshold the result
    입력 프레임에 Sobel 에지 검출 적용, 결과 임계값 지정
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None) -> dst
    #
    # • src: 입력 영상
    # • ddepth: 출력 영상 데이터 타입. -1이면 입력 영상과 같은 데이터 타입을 사용.
    # • dx: x 방향 미분 차수. 1차미분할지 2차미분 할지 결정
    # • dy: y 방향 미분 차수.
    # • dst: 출력 영상(행렬)
    # • ksize: 커널 크기. 기본값은 3.
    # • scale 연산 결과에 추가적으로 곱할 값. 기본값은 1.
    # • delta: 연산 결과에 추가적으로 더할 값. 기본값은 0.
    # • borderType: 가장자리 픽셀 확장 방식. 기본값은 cv2.BORDER_DEFAULT.
    
    # 엣지검출
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2) # sqrt는 제곱근
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255) # 1바이트 만큼 정수표현이 가능 (2^8)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY) # 이진화
    # threshold(img 이미지 , threshold_value 픽셀 문턱값 , value 픽셀 문턱값보다 클때 적용되는 최대값 , flag 문턱값 적용 방법 또는 스타일)

    return sobel_mag.astype(bool) # 데이터프레임의 타입 바꿔서 리턴


def get_binary_from_equalized_grayscale(frame): # 그레이스케일로 변환
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 변환

    eq_global = cv2.equalizeHist(gray) # 그레이스케일 이미지의 히스토그램화

    # 픽셀값(명도)이 250을 넘으면 255로 설정
    # _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    gray = cv2.medianBlur(gray, 15)
    th = cv2.adaptiveThreshold(gray, 230, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 7)

    return th


def binarize(img, verbose=False): # 이진화
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.
    입력 프레임을 차선 선을 최대한 강조하는 이진 이미지로 변환합니다.

    :param img: input color frame 입력 색상 프레임
    :param verbose: if True, show intermediate results 참이면 중간 결과 표시
    :return: binarized frame 이진화된 프레임
    """
    h, w = img.shape[:2]  # 1280 * 720
    # print("h : ",h)
    # print("w : ",w)

    binary = np.zeros(shape=(h, w), dtype=np.uint8)
    # np.zeros()는 주어진 shape나 type에 맞춰 0으로 채워진 array 반환

    # highlight yellow lines by threshold in HSV color space 옐로우 마스크
    # HSV_yellow_mask = thresh_frame_in_HSV(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    # binary = np.logical_or(binary, HSV_yellow_mask)

    # highlight white lines by thresholding the equalized frame 화이트 마스크
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask) # 둘중 하나가 true면 true

    # get Sobel binary mask (thresholded gradients) 엣지검출
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    # 이진 이미지의 "공백을 채우기 위해" 가벼운 형태를 적용하다.
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 0].set_facecolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        # ax[0, 2].imshow(HSV_yellow_mask, cmap='gray')
        # ax[0, 2].set_title('yellow mask')
        # ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing


if __name__ == '__main__':

    test_images = glob.glob('16cap/*.png')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binarize(img=img, verbose=True)

        # img_binarized = binarize(img=img, verbose=True)

        # cv2.imwrite('img/16_14-58-21_binarized.png', img_binarized)
