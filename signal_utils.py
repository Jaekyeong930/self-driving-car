import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# cv2.inrange 써도 가능


# 전방을 인식
# 전방 인식 영역 내 신호등이 있는가?
# 불이 켜진 신호등의 색이 감지되는가?
# 특정 범위 내의 값이면 RED 또는 GREEN으로 인식
# 그 신호등의

def get_green_pixels(frame, verbose=False):
    """
    Threshold a color frame in HSV space
    HSV 공간의 색상 프레임 임계값 지정
    """
    # 초록색 범위 지정
    green_HSV_th_min = np.uint8([35, 80, 80])
    green_HSV_th_max = np.uint8([165, 255, 255])

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # 색상 속성을 BGR에서 HSV로 변경

    green_out = cv2.inRange(HSV, green_HSV_th_min, green_HSV_th_max, dst=None)

    num_green_color = cv2.countNonZero(green_out)
    print('그린 픽셀의 수 : ', str(num_green_color))

    if verbose:
        plt.imshow(green_out, cmap='gray')
        plt.show()

    return num_green_color # 조건을 만족하는 픽셀값들


def get_red_pixels(frame, verbose=False):
    """
    Threshold a color frame in HSV space
    HSV 공간의 색상 프레임 임계값 지정
    """
    # 빨간색 범위 지정
    lower_red_HSV_th_min = np.array([0, 80, 80])
    lower_red_HSV_th_max = np.array([5, 255, 255])
    upper_red_HSV_th_min = np.array([170, 100, 100])
    upper_red_HSV_th_max = np.array([180, 255, 255])


    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 색상 속성을 BGR에서 HSV로 변경


    lower_red = cv2.inRange(HSV, lower_red_HSV_th_min, lower_red_HSV_th_max, dst=None)
    upper_red = cv2.inRange(HSV, upper_red_HSV_th_min, upper_red_HSV_th_max, dst=None)
    red_out = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
    # cv2.addWeighted(src1, alpha, src2, beta, gamma, dtype = None)은
    # 입력 이미지1(src1)에 대한 가중치1(alpha) 곱과 입력 이미지2(src2)에 대한 가중치2(beta) 곱의 합에 추가 합(gamma)

    num_red_color = cv2.countNonZero(red_out)
    print('레드 픽셀의 수 : ', str(num_red_color))

    if verbose:
        plt.imshow(red_out, cmap='gray')
        plt.show()

    return num_red_color  # 조건을 만족하는 픽셀값들



def camera_cap() :
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()

        if (ret):
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()




if __name__ == "__main__":

    mode = 'img'

    if mode == 'video' :
        # 카메라 입력
        cap = cv2.VideoCapture(0)

        while (True):
            ret, frame = cap.read()

            if (ret):
                cv2.imshow('frame', frame)

                green_output = get_green_pixels(frame)
                red_output = get_red_pixels(frame)

                now_signal = None  # 현재 신호등 색 판별할 변수

                # 해당 색을 나타내는 픽셀의 수로 신호 판별
                if green_output > red_output:
                    now_signal = 'green'
                else:
                    now_signal = 'red'

                print('>> 현재 신호등은 ', now_signal)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    else : # 이미지
        test_images = glob.glob('signal_images/*.png')

        for test_image in test_images:
            print()
            print('파일명 :', str(test_image))

            frame = cv2.imread(test_image)

            # 영역 지정
            # frame = img[100:600, 200:700].copy()

            green_output = get_green_pixels(frame, verbose=True)
            red_output = get_red_pixels(frame, verbose=True)

            now_signal = None  # 현재 신호등 색 판별할 변수

            # 해당 색을 나타내는 픽셀의 수로 신호 판별
            if green_output > red_output:
                now_signal = 'green'
            else:
                now_signal = 'red'

            print('>> 현재 신호등은 ', now_signal)




        # img_binarized = binarize(img=img, verbose=True)

        # cv2.imwrite('img/16_14-58-21_binarized.png', img_binarized)





