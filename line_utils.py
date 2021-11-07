# 차선 검출

import numpy as np
import math
import cv2
import glob
import collections
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from globals import ym_per_pix, xm_per_pix


print("=================================line_util.py=================================")

class Line:
    """
    Class to model a lane-line. 차선에 대한 모델
    """
    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        # 마지막 반복이 감지된 경우 표시할 플래그
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        # 마지막 반복에 적합된 다항식 계수
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        # 마지막 N회 반복의 다항식 계수 리스트
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len) # 최근 감지된 픽셀 저장 / 최대 10회 반복
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len) # 최근 감지된 ? / 최대 20회 반복

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        # 검색된 선의 모든 픽셀좌표 저장
        self.all_x = None
        self.all_y = None

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.

        :param new_fit_pixel: new polynomial coefficients (pixel) 새 다항식 계수
        :param new_fit_meter: new polynomial coefficients (meter) 새 다항식 계수
        :param detected: if the Line was detected or inferred 라인이 감지되거나 유추된 경우
        :param clear_buffer: if True, reset state 참일 경우 상태 재설정
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel # 새 다항식 계수 (pixel)
        print(self.last_fit_pixel)
        self.last_fit_meter = new_fit_meter # 새 다항식 계수 (meter)
        print(self.last_fit_meter)

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)


    #인식된 차선 그리기
    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        # x와 y 포인트를 cv2.fillPoly()에 사용할 수 있는 형식으로 다시 캐스트하는 방법
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)

#numpy.mean(a 평균을 원하는 숫자를 포함하는 배열, axis=None 평균이 계산되는 축, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)[source]

    @property
    # average of polynomial coefficients of the last N iterations 최근 N회 반복의 다항식 계수 평균
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # radius of curvature of the line (averaged) 선의 곡률 반지름 (평균값)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit # 계수
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


    '''
    이전 차선 선 감지 단계에서 2차 다항식이 np.polyfit()을 사용하여 각 차선 선에 적합된다. 
    이 함수는 곡선을 설명하는 3개의 계수, 즉 2차 항과 1차 항 모두의 계수에 편향을 더한 계수를 반환합니다. 
    이 방정식에 따라 이 계수로부터 곡선의 곡률 반경을 계산할 수 있습니다.
    '''
    @property
    # radius of curvature of the line (averaged) 곡률반경 (평균)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0) # 계수 = np.mean(
        # np.mean 은 주어진 배열의 산술평균을 계산 (주어진 수의 합을 수의 개수로 나누는 일반적인 평균)
        # axis 는 산술평균이 계산되는 축, 0은 열을 따라 계산, 1은 행을 따라 계산, 없으면 다차원 배열을 평면화된 목록으로 취급

        #곡률 = 휘어진 정도 / 곡률반경 = 곡선을 이루는 원의 반지름
        #곡률의 역수 = 곡률 반경 // 곡률이 클수록 곡률 반경이 작고 곡률이 작을수록 곡률 반경이 크다

        for i in coeffs :
            print('coeffes : ', i)

        # Radius of curvature(곡률반경) 공식에 따라 리턴
        # (2 * coeffs[0] * y_eval + coeffs[1]) = tanΘ = y'
        # np.absolute(2 * coeffs[0]) = y''
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])



def get_fits_by_sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.
    이진 이미지에서 감지된 차선선에 대한 다항식 계수를 가져옵니다.

    :param birdeye_binary: input bird's eye view binary image 조감도 이진 이미지 입력
    :param line_lt: left lane-line previously detected 이전에 감지된 좌측 차선
    :param line_rt: left lane-line previously detected 이전에 감지된 우측 차선
    :param n_windows: number of sliding windows used to search for the lines 선을 검색하는 데 사용되는 슬라이딩 창의 수
    :param verbose: if True, display intermediate output 참이면 중간 출력을 표시합니다.
    :return: updated lane lines and output image 업데이트된 차선 선 및 출력 이미지
    """
    height, width = birdeye_binary.shape # 이미지의 가로 세로 사이즈 획득

    # Assuming you have created a warped binary image called "binary_warped" "binary_warped"라는 뒤틀린 이진 이미지를 만들었다고 가정합니다.
    # Take a histogram of the bottom half of the image 이미지 아래쪽의 히스토그램을 취합니다.
    histogram = np.sum(birdeye_binary[height//2:-30, :], axis=0)

    # Create an output image to draw on and  visualize the result 결과를 그리고 시각화할 출력 이미지 만들기
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    # Find the peak of the left and right halves of the histogram 히스토그램의 왼쪽과 오른쪽 절반의 피크 찾기
    # These will be the starting point for the left and right lines 왼쪽과 오른쪽 선의 시작점이 됩니다.
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows 창 높이 설정
    window_height = np.int64(height / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image 영상에서 0이 아닌 모든 픽셀의 x 및 y 위치 식별
    # nonzero : 0이 아닌 요소의 인덱스를 반환
    nonzero = birdeye_binary.nonzero() # 흑백 이미지에서 1인 픽셀의 인덱스 추출
    print("nonzero : ", nonzero)
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window 각 창에 대해 업데이트할 현재 위치
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100  # width of the windows +/- margin  창의 너비 +/- 여백
    minpix = 50   # minimum number of pixels found to recenter window 최신 창에서 발견된 최소 픽셀 수

    # Create empty lists to receive left and right lane pixel indices 왼쪽 및 오른쪽 차선 픽셀 색인을 수신할 빈 리스트 작성
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows): # n_windows 값이 클 수록 차선을 더 잘게 나누어서 인식
        # Identify window boundaries in x and y (and right and left)  x와 y(오른쪽과 왼쪽)에서 창 경계 식별
        win_y_low = height - (window + 1) * window_height # 높이
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image 창에 시각화 이미지를 그림
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window 창 내에서 x와 y의 0이 아닌 픽셀 식별
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists 이 인덱스를 목록에 추가
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        # 만약 minpix 픽셀을 찾은 경우 평균 위치에서 다음 창을 다시 시작합니다
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices 인덱스 배열 연결
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions 왼쪽 및 오른쪽 선 픽셀 위치 추출

    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    print("line_lt.all_x : ", line_lt.all_x , "line_lt.all_y : ", line_lt.all_y)
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
    print("line_rt.all_x : ", line_rt.all_x, "line_rt.all_y : ", line_rt.all_y)

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    print("left_fit_fixel : ", left_fit_pixel, " / left_fit_meter : ", left_fit_meter)
    print("right_fit_fixel : ", right_fit_pixel, " / right_fit_meter : ", right_fit_meter)
    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting 그림을 그릴 x 및 y 값 생성
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0] #
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    if verbose:
        f, ax = plt.subplots(1, 2)
        f.set_facecolor('white')
        ax[0].imshow(birdeye_binary, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='yellow')
        ax[1].plot(right_fitx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return line_lt, line_rt, out_img


def get_fits_by_previous_fits(birdeye_binary, line_lt, line_rt, verbose=False):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.
    This function starts from previously detected lane-lines to speed-up the search of lane-lines in the current frame.
    이진 이미지에서 감지된 차선선에 대한 다항식 계수를 가져옵니다.
    이 기능은 이전에 감지된 차선선에서 시작하여 현재 프레임에서 차선 검색 속도를 높입니다.

    :param birdeye_binary: input bird's eye view binary image 이진이미지 조감도 입력
    :param line_lt: left lane-line previously detected 이전에 감지된 왼쪽 차선
    :param line_rt: left lane-line previously detected 이전에 감지된 오른쪽 차선
    :param verbose: if True, display intermediate output 만약 참이면 중간출력 표시
    :return: updated lane lines and output image 업데이트된 차선 선 및 출력 이미지
    """

    height, width = birdeye_binary.shape

    left_fit_pixel = line_lt.last_fit_pixel
    right_fit_pixel = line_rt.last_fit_pixel

    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
    (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
    nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
    right_lane_inds = (
    (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
    nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

    # Extract left and right line pixel positions 왼쪽 및 오른쪽 선 픽셀 위치 추출
    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    # numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)[source] 선의 다항식을 찾아냄냄
    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting 그림을 그릴 x 및 y 값 생성
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # Create an image to draw on and an image to show the selection window 그릴 이미지와 선택 창을 표시할 이미지를 만듭니다.
    img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels 색(왼쪽 및 오른쪽 선 픽셀)
    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area 검색 창 영역을 나타내는 폴리곤 생성
    # And recast the x and y points into usable format for cv2.fillPoly() 그리고 x와 y 포인트를 cv2.fillPoly()에 사용할 수 있는 형식으로 다시 캐스트
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image 뒤틀린 빈 이미지 위에 차선을 그립니다.
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    if verbose:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return line_lt, line_rt, img_fit


# 인식된 차선 그리기
def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
    """
    Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.
    주행 가능한 차선 영역과 감지된 차선 선을 원래(왜곡되지 않은) 프레임에 모두 그립니다.
    :param img_undistorted: original undistorted color frame 왜곡되지 않은 원본 색 프레임
    :param Minv: (inverse) perspective transform matrix used to re-project on original frame 원래 프레임에 재투영하는 데 사용되는 (원근 변환 매트릭스)
    :param line_lt: left lane-line previously detected
    :param line_rt: right lane-line previously detected
    :param keep_state: if True, line state is maintained
    :return: color blend
    """
    height, width, _ = img_undistorted.shape
    print("height ", height)
    print("width ", width)
    print("_ ", _)

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them 이제 그것들을 강조하기 위해 실선을 따로 그린다.
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)

    return blend_onto_road

######################## 메인함수 ########################
if __name__ == '__main__':

    # 차선객체 생성
    line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # show result on test images
    for test_img in glob.glob('16cap/*.png'):

        # 이미지 읽기
        img = cv2.imread(test_img)
        
        # 이미지 왜곡 보정
        img_undistorted = undistort(img, mtx, dist, verbose=False)
        # 이미지 이진화
        img_binary = binarize(img_undistorted, verbose=False)
        # 이미지 원근 보정
        img_birdeye, M, Minv = birdeye(img_binary, verbose=False)
        # 선 인식
        line_lt, line_rt, img_out = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=7, verbose=True)








