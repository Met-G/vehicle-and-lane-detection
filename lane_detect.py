import cv2
import numpy as np


def get_edge_img(color_img, gaussian_ksize=5, gaussian_sigmax=1,
                 canny_threshold1=250, canny_threshold2=350):
    """
    灰度化,模糊,canny变换,提取边缘
    :param color_img: 彩色图,channels=3
    """
    gaussian = cv2.GaussianBlur(color_img, (gaussian_ksize, gaussian_ksize),
                                gaussian_sigmax)
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


def roi_mask(gray_img):
    """
    对gray_img进行掩膜
    :param gray_img: 灰度图,channels=1
    """
    poly_pts = np.array([[[420, 850], [950, 550], [1000, 550], [1600, 850]]])
    mask = np.zeros_like(gray_img)
    mask = cv2.fillPoly(mask, pts=poly_pts, color=255)
    img_mask = cv2.bitwise_and(gray_img, mask)
    return img_mask


def get_lines(edge_img):
    """
    获取edge_img中的所有线段
    :param edge_img: 标记边缘的灰度图
    """

    def calculate_slope(line):
        """
        计算线段line的斜率
        :param line: np.array([[x_1, y_1, x_2, y_2]])
        :return:
        """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)

    def reject_abnormal_lines(lines, threshold=0.2):
        """
        剔除斜率不一致的线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        """
        slopes = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slopes)
            diff = [abs(s - mean) for s in slopes]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slopes.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines

    def least_squares_fit(lines):
        """
        将lines中的线段拟合成一条线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
        """
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        poly = np.polyfit(x_coords, y_coords, deg=1)
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
        return np.array([point_min, point_max], dtype=np.int)

    # 获取所有线段
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40,
                            maxLineGap=20)
    # 按照斜率分成车道线
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]
    # 剔除离群线段
    left_lines = reject_abnormal_lines(left_lines)
    right_lines = reject_abnormal_lines(right_lines)

    return least_squares_fit(left_lines), least_squares_fit(right_lines)


def draw_lines(img, lines):
    """
    在img上绘制lines
    :param img:
    :param lines: 两条线段: [np.array([[xmin1, ymin1], [xmax1, ymax1]]), np.array([[xmin2, ymin2], [xmax2, ymax2]])]
    :return:
    """
    left_line, right_line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255),
             thickness=5)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]),
             color=(0, 255, 255), thickness=5)


def show_lane(color_img):
    """
    在color_img上画出车道线
    :param color_img: 彩色图,channels=3
    :return:
    """
    edge_img = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img)
    lines = get_lines(mask_gray_img)
    draw_lines(color_img, lines)
    return color_img


cap = cv2.VideoCapture('mi_1.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
outVideo = cv2.VideoWriter('mi_1_lanes.avi', fourcc, fps, (width, height))
print(fps, width, height)

while True:

    ret, frame = cap.read()
    if not ret:
        print("... end of video file reached")
        break

    if frame is None:
        break

    raw = frame.copy()
    result = show_lane(frame)
    cv2.namedWindow('frame', 0)
    cv2.imshow('frame', result)
    cv2.waitKey(1)
    outVideo.write(result)

cap.release()
cv2.destroyAllWindows()
