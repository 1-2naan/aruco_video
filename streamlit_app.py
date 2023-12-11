import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from cv2 import aruco

# Pose estimation function
def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, count, tvec_store, time_store, frame_store):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict_type, parameters=parameters)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, matrix_coefficients, distortion_coefficients)
            frame_store = np.vstack((frame_store, count))
            tvec_store = np.vstack((tvec_store, tvec[0][0]))
            time_store = np.vstack((time_store, np.array(time.perf_counter())))
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, tvec_store, time_store, frame_store

# Streamlit UI
st.title('Video Upload for Pose Estimation')

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi"])
if uploaded_file is not None:
    # Progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    tvec_store = np.empty((0, 3))
    time_store = np.empty((0, 1))
    frame_store = np.empty((0, 1))
    c = 0

    aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
    k = np.load('calibration_matrix.npy')
    d = np.load('distortion_coefficients.npy')

    video = cv2.VideoCapture(uploaded_file.name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read()
        c += 1

        if not ret:
            break

        frame, tvec_store, time_store, frame_store = pose_esitmation(frame, aruco_dict_type, k, d, c, tvec_store, time_store, frame_store)

        # Update progress bar and status
        progress_bar.progress(c / total_frames)
        status_text.text(f'Processing frame {c}/{total_frames}')

    arr = np.hstack((tvec_store, time_store, frame_store))
    df = pd.DataFrame(arr, columns=['tvec_x', 'tvec_y', 'tvec_z', 'time', 'frame'])
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=uploaded_file.name.split('.')[0] + '.csv',
        mime='text/csv',
    )

    video.release()

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()
