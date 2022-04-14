import cv2
import numpy as np
from vidstab import VidStab, download_ostrich_video, layer_overlay

# # Download test video to stabilize
# if not os.path.isfile("ostrich.mp4"):
#     download_ostrich_video("ostrich.mp4")

# Initialize object tracker, stabilizer, and video reader
object_tracker = cv2.TrackerCSRT_create()
stabilizer = VidStab()
vidcap = cv2.VideoCapture("data\\15_meter.MOV")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = False

# Initialize bounding box for drawing rectangle around tracked object
object_bounding_box = None

box_dim = [0,0]
box = [0, 0, 4000, 2250]

f = 0

while True:
    f = f + 1
    print(f)
    grabbed_frame, frame = vidcap.read()

    #frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=20, border_size=0)

    frame = False

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Draw rectangle around tracked object if tracking has started
    if object_bounding_box is not None:
        success, object_bounding_box = object_tracker.update(stabilized_frame)

        if success:
            (x, y, w, h) = [int(v) for v in object_bounding_box]
            if box_dim[0] == 0:
                box_dim[0] = w
                box_dim[1] = h
            
            center = [x + (w / 2), y + (h / 2)]

            box = [int(center[0] - (box_dim[0] / 2)), int(center[1] - (box_dim[1] / 2)), box_dim[0], box_dim[1]]

            if box[3] / box[2] > 9 / 16:
                #Alter box 2
                ratio_width = int((box[3] / 9) * 16)
                box[2] = ratio_width
                box[0] = box[0] - int(ratio_width * 0.25)
            else:
                #alter box 3
                ratio_height = int((box[2] / 16) * 9)
                box[3] = ratio_height
                box[1] = box[1] - int(ratio_height * 0.25)

            stabilized_frame = stabilized_frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            #print(box[2], box[3])
            #cv2.rectangle(stabilized_frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
            if not out:
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter('output.mp4',fourcc, fps, (box[2],box[3]))
            #stabilized_frame = cv2.resize(stabilized_frame,(box[2],box[3]),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(stabilized_frame)

    # Display stabilized output
    cv2.imshow("Frame", stabilized_frame)
    

    key = cv2.waitKey(5)

    # Select ROI for tracking and begin object tracking
    # Non-zero frame indicates stabilization process is warmed up
    if stabilized_frame.sum() > 0 and object_bounding_box is None:
        object_bounding_box = cv2.selectROI(
            "Frame", stabilized_frame, fromCenter=False, showCrosshair=True
        )
        object_tracker.init(stabilized_frame, object_bounding_box)
    elif key == 27:
        break

vidcap.release()
out.release()
cv2.destroyAllWindows()
