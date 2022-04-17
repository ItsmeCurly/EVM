import cv2

vidcap = cv2.VideoCapture("68_bpm_output.mp4")     # The video stream
fourcc = cv2.VideoWriter_fourcc(*'XVID') # The video writer file format

out = False

dim = [0,0]

while True:
    grabbed_frame, frame = vidcap.read()            # Read a frame of the video

    if frame is None:
        break

    shape = frame.shape

    if dim[0] == 0:
        dim[0] = shape[0]
        dim[1] = shape[1]
        if dim[0] % 8 != 0:
            dim[0] = dim[0] + (8 - (dim[0] % 8))

        if dim[1] % 8 != 0:
            dim[1] = dim[1] + (8 - (dim[1] % 8))

    #frame = cv2.resize(frame,(int(dim[1] / 2), int(dim[0] / 2)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

    frame = cv2.pyrUp(frame)
    #frame = cv2.pyrUp(frame)

    shape = frame.shape

    if not out:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('face_output.avi',fourcc, fps, (shape[1],shape[0]))

    out.write(frame)

vidcap.release()
out.release()