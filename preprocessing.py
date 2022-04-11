import cv2
import numpy as np

# Open the video
cap = cv2.VideoCapture("data\\flir_1.mp4")

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(w_frame, h_frame)

# Here you can define your croping values
x, y, h, w = 216, 216, 96, 200

# output
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("result.avi", fourcc, fps, (w, h))


# Now we start
while cap.isOpened():
    ret, frame = cap.read()

    cnt += 1  # Counting frames

    # Avoid problems when video finish
    if ret:
        # Croping the frame
        crop_frame = frame[y : y + h, x : x + w]

        # Percentage
        xx = cnt * 100 / frames
        # print(int(xx), "%")

        # Saving from the desired frames
        # if 15 <= cnt <= 90:
        #    out.write(crop_frame)

        # I see the answer now. Here you save all the video

        # Just to see the video in real time

        img_thresh = cv2.inRange(
            cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV), (165, 0, 0), (255, 255, 255)
        )
        img_thresh2 = cv2.inRange(
            cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV), (0, 0, 0), (25, 255, 255)
        )
        img_thresh3 = cv2.inRange(
            cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV), (0, 0, 0), (255, 16, 255)
        )

        # contours, hierarchy = cv2.findContours(
        # img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(img_thresh, contours, -1, (0,255,0), 3)
        cv2.imshow("frame", frame)
        cv2.imshow(
            "img_thresh", cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh)
        )
        cv2.imshow(
            "img_thresh2", cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh2)
        )
        cv2.imshow(
            "img_thresh3", cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh3)
        )
        cv2.imshow("cropped", cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV))

        cropped_frame = (
            cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh)
            + cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh2)
            + cv2.bitwise_and(crop_frame, crop_frame, mask=img_thresh3)
        )

        out.write(cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
