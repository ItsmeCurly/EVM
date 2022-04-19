import cv2

vidcap = cv2.VideoCapture("1080p_output.mp4")     # The video stream
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # The video writer file format

out = False

while True:
    grabbed_frame, frame = vidcap.read()            # Read a frame of the video

    if frame is None:
        break

    frame = cv2.pyrUp(frame)

    shape = frame.shape

    if not out:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('1080p_output2.mp4',fourcc, fps, (shape[1],shape[0]))

    out.write(frame)

vidcap.release()
out.release()
