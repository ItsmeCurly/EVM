import cv2
import numpy as np
from vidstab import VidStab, layer_overlay

object_tracker = cv2.TrackerCSRT_create()           # Initialize the object tracker
primary_stabilizer = VidStab()                      # Initialize the video stabilizer pre-tracker
secondary_stabilizer = VidStab()                    # Initialize the video stabilizer post-tracker
vidcap = cv2.VideoCapture("data\\1080p.mp4")     # The video stream
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # The video writer file format
out = False                                         # The video writer (to be initialized once the framerate is known)
object_bounding_box = None                          # Initialize the bounding box

box_dim = [0, 0]                                    # The chosen dimension of th bounding box

while True:
    grabbed_frame, frame = vidcap.read()            # Read a frame of the video

    stabilized_frame = primary_stabilizer.stabilize_frame(input_frame=frame, border_size=0)# Pre-stabilization

    if stabilized_frame is None:
        break

    if object_bounding_box is not None:             # If the bounding box has been drawn
        success, object_bounding_box = object_tracker.update(stabilized_frame)# Update the position of the bounding box

        if success:
            (x, y, w, h) = [int(v) for v in object_bounding_box] # The bounding box location data

            # Set the initial bounding box dimensions if they have not already ben set
            if box_dim[0] == 0:
                dim = max(w, h) + 8
                if dim % 8 != 0:
                    dim = dim + (8 - (dim % 8))
                box_dim[0] = dim
                box_dim[1] = dim
    
            # Place the bounding box in the center of the selected area
            center = [x + (w / 2), y + (h / 2)]
            box = [int(center[0] - (box_dim[0] / 2)), int(center[1] - (box_dim[1] / 2)), box_dim[0], box_dim[1]]

            # Crop to the inside of the bounding box
            stabilized_frame = stabilized_frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

            # If a frame was created, save it.
            if stabilized_frame.sum() > 0:

                # Perform secondary stabilization to account for shaky object tracking.
                final_frame = secondary_stabilizer.stabilize_frame(input_frame=stabilized_frame, border_size=-8)

                # If secondary stabilization returns a frame (Remember that it starts with 30 blank frames, so between primary and secondary we lose 1 to 2 seconds from the start).
                if final_frame.sum() > 0:
                    # If the writer has not been initialized, initialize it to the shape of the outputted frames.
                    if not out:
                        fps = vidcap.get(cv2.CAP_PROP_FPS)
                        shape = final_frame.shape
                        out = cv2.VideoWriter('1080p_output.mp4',fourcc, fps, (shape[1],shape[0]))
                    #Write the output, and make sure it is displayed on the user interface.
                    #final_frame = cv2.fastNlMeansDenoisingColored(final_frame,None,5,5,6,16)
                    out.write(final_frame)
                    stabilized_frame = final_frame

    # Display the image on the UI. Note that this may or may not include the secondary stabilized image.
    cv2.imshow("Frame", stabilized_frame)

    # Below is code for creating the bounding box.
    key = cv2.waitKey(5)
    # Select ROI for tracking and begin object tracking
    # Non-zero frame indicates stabilization porcess is warmed up
    if stabilized_frame.sum() > 0 and object_bounding_box is None:
        object_bounding_box = cv2.selectROI(
            "Frame", stabilized_frame, fromCenter=False, showCrosshair=True
        )
        object_tracker.init(stabilized_frame, object_bounding_box)
    elif key == 27:
        break


# Close all of the files.
vidcap.release()
out.release()
cv2.destroyAllWindows()