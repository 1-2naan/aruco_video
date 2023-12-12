import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from cv2 import aruco
import io

def calculate_metrics(data, peaks):
    speeds = [0]  # Starting with an initial average speed of 0
    velocities = [0]  # Starting with 0 velocity
    accelerations = [0]  # Starting with 0 acceleration
    
    for i in range(1, len(peaks)):
        # Average Speed calculation
        distance_diff = data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]
        time_diff = data['Timestamp'].iloc[peaks[i]] - data['Timestamp'].iloc[peaks[i-1]]
        average_speed = distance_diff / time_diff if time_diff else 0
        speeds.append(average_speed)
        
        # Instantaneous Velocity calculation
        velocity = (data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]) / time_diff if time_diff else 0
        velocities.append(velocity)
        
        # Acceleration calculation
        acceleration = (velocities[i] - velocities[i-1]) / time_diff if i > 1 and time_diff else 0
        accelerations.append(acceleration)
        
    return speeds, velocities, accelerations

def plot_hand_movement(data, max_width, min_width):
    y = data['Distance'].values
    timestamps = data['Timestamp'].values
    median_val = np.median(y)
    
    max_peaks, _ = find_peaks(-y, width=max_width)
    min_peaks, _ = find_peaks(y, width=min_width)
    
    # Filter peaks using the median value
    filtered_max_peaks = max_peaks[y[max_peaks] < median_val]
    filtered_min_peaks = min_peaks[y[min_peaks] > median_val * 1.5]  # Adjust this multiplier as needed

    # Combine and sort all peaks
    all_peaks = np.sort(np.concatenate((filtered_max_peaks, filtered_min_peaks)))

    # Calculate average speed, velocities, and accelerations
    speeds, velocities, accelerations = calculate_metrics(data, all_peaks)

    # Create a DataFrame to download
    peaks_df = pd.DataFrame({
        'Timestamp': data['Timestamp'][all_peaks],
        'Distance': data['Distance'][all_peaks],
        'Average Speed': speeds,
        'Velocity': velocities,
        'Acceleration': accelerations
    })

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y, label='Hand Movement')
    plt.plot(timestamps[filtered_max_peaks], y[filtered_max_peaks], "bo", label='Maximum Peaks')
    plt.plot(timestamps[filtered_min_peaks], y[filtered_min_peaks], "ro", label='Minimum Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Distance Moved')
    plt.title('Hand Movement with Velocity and Acceleration at Peaks')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

    return peaks_df
    
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
            #st.text(f"Markers detected in frame: {c}")
            #st.write("4")
    else:
        #st.text(f"No markers detected in frame: {c}")
        #st.write("5")
    return frame, tvec_store, time_store, frame_store

# Streamlit UI
st.title('Video Upload for Pose Estimation')

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi"])
if uploaded_file is not None:
    #st.write("1")
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
    #st.write("matrix,coeff:",k)
    #st.write("dist:,",d)
    #st.write("2")
    #video = cv2.VideoCapture(uploaded_file.name)
    # To read the file, you can use BytesIO
    #file_bytes = io.BytesIO(uploaded_file.read())
    
    # Now use the file_bytes to open the video in OpenCV
    #video = cv2.VideoCapture(file_bytes)

    # Define a path for the video file
    video_file_path = 'temp_video.mp4'  # or use a unique naming scheme

    # Write the uploaded file to the defined path
    with open(video_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Now use the path with OpenCV
    video = cv2.VideoCapture(video_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read()
        c += 1

        if not ret:
            st.write("3")
            break

        frame, tvec_store, time_store, frame_store = pose_esitmation(frame, aruco_dict_type, k, d, c, tvec_store, time_store, frame_store)

        # Update progress bar and status
        progress_bar.progress(c / total_frames)
        status_text.text(f'Processing frame {c}/{total_frames}')

    arr = np.hstack((tvec_store, time_store, frame_store))
    df = pd.DataFrame(arr, columns=['tvec_x', 'tvec_y', 'tvec_z', 'time', 'frame'])

    df.columns=['X','Y','Z','T','Frame_num']
    df[['Timestamp']]=df[['Frame_num']]/60
    df['X1']=df['X']-df['X'][0]
    df['Y1']=df['Y']-df['Y'][0]
    df['Z1']=df['Z']-df['Z'][0]
    df['Distance'] = (df['X1']**2+df['Y1']**2+df['Z1']**2)**0.5
    df2=df[['Timestamp','Distance']]
    
    #csv = df.to_csv(index=False).encode('utf-8')
    # Sliders to adjust width for peak finding for each file
    st.write("Adjust the peak width settings for this file:")
    max_width = st.slider(f'Select Maximum Peak Width for {uploaded_file.name}', min_value=1, max_value=200, value=30, key=f"max_width_{uploaded_file.name}")
    min_width = st.slider(f'Select Minimum Peak Width for {uploaded_file.name}', min_value=1, max_value=200, value=40, key=f"min_width_{uploaded_file.name}")

    # Process the file and plot the data
    peaks_df = plot_hand_movement(df2, max_width, min_width)

   ''' st.download_button(
        label="Download CSV",
        data=csv,
        file_name=uploaded_file.name.split('.')[0] + '.csv',
        mime='text/csv',
    )'''

# Provide a download link for the peaks DataFrame
    st.download_button(
        label="Download Peaks Data as CSV",
        data=peaks_df.to_csv(index=False).encode('utf-8'),
        file_name=f'{uploaded_file.name.split(".")[0]}_peaks_data.csv',
        mime='text/csv',
    )

    video.release()

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()
