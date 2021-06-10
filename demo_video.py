from lib.detector import VehicleDetector

import cv2

cap = cv2.VideoCapture('./video_lanes.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
outVideo = cv2.VideoWriter('video_detected.avi', fourcc, fps, (width, height))
print(fps, width, height)

det = VehicleDetector()

cnt = 0
while True:

    ret, frame = cap.read()
    if not ret:
        print("... end of video file reached")
        break

    if frame is None:
        break

    raw = frame.copy()

    result = det.detect(frame)

    cv2.namedWindow('frame', 0)
    cv2.imshow('frame', result)
    cv2.waitKey(1)

    outVideo.write(result)
    cnt = cnt + 1

cap.release()
cv2.destroyAllWindows()
