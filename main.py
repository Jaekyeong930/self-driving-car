# 아나콘다 환경
# 파이썬 3.8.5
# openCV 4.5.1


import cv2
import math
import os # Operating System의 약자로서 운영체제에서 제공되는 여러 기능을 파이썬에서 수행
import matplotlib.pyplot as plt # 데이터 시각화
from calibration_utils import calibrate_camera, undistort # 캘리변환
from binarization_utils import binarize # 이진변환
from perspective_utils import birdeye # 원근변환
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from moviepy.editor import VideoFileClip # 영상처리
import numpy as np # 복잡한 수식 계산
from globals import xm_per_pix, time_window


# verbose = 과정샷 띄울건지 말건지

processed_frames = 0                    # counter of frames processed (when processing video) 가공된 프레임 수
line_lt = Line(buffer_len=time_window)  # line on the left of the lane 차선 왼쪽 라인
line_rt = Line(buffer_len=time_window)  # line on the right of the lane 차선 오른쪽 라인


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images
    모든 중간 파이프라인 이미지를 고려하여 최종 출력 혼합 준비

    :param blend_on_road: color image of lane blend onto the road 도로+차선 혼합 컬러이미지
    :param img_binary: thresholded binary image 임계값 이진 이미지
    :param img_birdeye: bird's eye view of the thresholded binary image 임계 이진 이미지의 조감도
    :param img_fit: bird's eye view with detected lane-lines highlighted 감지된 차선의 선이 강조표시된 조감도
    :param line_lt: detected left lane-line 감지된 왼쪽 차선
    :param line_rt: detected right lane-line 감지된 오른쪽 차선
    :param offset_meter: offset from the center of the lane 차선 중앙에서 벗어남
    :return: pretty blend with all images and stuff stitched 리턴값
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2 # 썸네일 비율
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area 상단 영역을 강조 표시하기 위해 그레이 사각형 추가
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image 이진 이미지 썸네일 추가
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view 조감도 썸네일 추가
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted) 차선이 강조된 썸네일 추가
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend 오른쪽 상단에 변환&오프셋 정보 추가
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter]) # ~의 산술평균 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 화면에 출력
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}cm'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road

# 오프셋 계산 (유추된 차선 값 반환)
def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.

    :param line_lt: detected left lane-line 감지된 왼쪽 차선
    :param line_rt: detected right lane-line 감지된 오른쪽 차선
    :param frame_width: width of the undistorted frame 왜곡되지 않은 프레임 폭
    :return: inferred offset 유추된 상계값 반환
    """
    # 현재 차선인식 프로세싱을 돌렸을때 프레임 중앙값과 차선의 중앙값의 위 왼쪽라인 지점에서
    # 레인너비의 반을 더한것에서 인식 프레임의 중심만큼 뺀다. 그게 프레임의 중심과 차선의 중심의 오차이다.
    # 이를 이용해서 서보모터를 얼마나 제어할 것인지?


    if line_lt.detected and line_rt.detected: # 왼쪽 차선과 오른쪽 차선이 인식되었다면 곡률반경 인식
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2 # 중앙지점은 프레임 너비/2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint) # (왼쪽라인+레인너비/2) - 프레임의 중심
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def process_pipeline(frame, keep_state=True): # 인식된 차선 그리기
    """
    Apply whole lane detection pipeline to an input color frame.
    모든 인식된 차선을 컬러프레임으로 적용
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    # 보정에 있는 계수를 사용해 영상의 정렬을 해제
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    # 프레임 이진화, 가능한 많이 차선라인을 강조표시
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    # 조감도 획득을 위한 원근법 변환 계산
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    # 발견된 차선에 2차 다항식 곡선 적용
    # 주어진 이진 이미지의 어떤 픽셀이 레인 라인에 속하는지 식별
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    # 차선 중앙에서 미터 단위로 오프셋 계산
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    # 차선으로 둘러싸인 표면을 원래의 프레임에 다시 그림
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    # 파이프라인의 다른 단계에서 최종 출력 이미지 상단에 스티치
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    return blend_output

###################################### 메인 실행 함수 ######################################
if __name__ == '__main__':

    # first things first: calibrate the camera
    # 첫번째로 할것 : 카메라 보정
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    mode = 'images'

    # 영상 차선인식
    if mode == 'video':
        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
        clip.write_videofile('out_{}_{}.mp4'.format(selector, time_window), audio=False)

    # 사진 차선인식
    else:
        test_img_dir = 'curvature_test_img' # 차선인식을 할 폴더
        # 테스트 이미지 수 만큼 반복
        for test_img in os.listdir(test_img_dir):

            # 이미지 읽기
            frame = cv2.imread(os.path.join(test_img_dir, test_img))

            # 프로세싱 과정 합치기
            blend = process_pipeline(frame, keep_state=False)

            # 프로세싱한 이미지들을 아웃풋 폴더에 저장
            cv2.imwrite('output_images/{}'.format(test_img), blend)

            # 실제 그림을 표시
            plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            plt.show()
