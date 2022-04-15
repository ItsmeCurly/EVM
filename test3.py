import os

import cv2
from vidstab import VidStab, download_ostrich_video, layer_overlay

# # Download test video to stabilize
# if not os.path.isfile("ostrich.mp4"):
#     download_ostrich_video("ostrich.mp4")

# Initialize object tracker, stabilizer, and video reader
object_tracker = cv2.TrackerCSRT_create()
stabilizer = VidStab()
vidcap = cv2.VideoCapture("output.mp4")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = False
# Initialize bounding box for drawing rectangle around tracked object
object_bounding_box = None

while True:
    grabbed_frame, frame = vidcap.read()

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=0)

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Draw rectangle around tracked object if tracking has started
    #if object_bounding_box is not None:
    #    success, object_bounding_box = object_tracker.update(stabilized_frame)

    #    if success:
    #        (x, y, w, h) = [int(v) for v in object_bounding_box]
    #        cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display stabilized output
    cv2.imshow("Frame", stabilized_frame)
    shape = stabilized_frame.shape
    if not out:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output_stabilized.mp4',fourcc, fps, (shape[1],shape[0]))
    #stabilized_frame = cv2.resize(stabilized_frame,(shape[1],shape[0]),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    if stabilized_frame.sum() > 0:
        out.write(stabilized_frame)

    key = cv2.waitKey(5)

    # Select ROI for tracking and begin object tracking
    # Non-zero frame indicates stabilization process is warmed up
    #if stabilized_frame.sum() > 0 and object_bounding_box is None:
        #object_bounding_box = cv2.selectROI(
        #    "Frame", stabilized_frame, fromCenter=False, showCrosshair=True
        #)
        #object_tracker.init(stabilized_frame, object_bounding_box)
    #elif key == 27:
    #    break

vidcap.release()
cv2.destroyAllWindows()