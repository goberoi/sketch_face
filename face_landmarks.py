import face_recognition
import cv2
import pprint
import numpy as np

# Settings
process_nth_frame = 2
small_frame = False

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_landmarks_list = []
frame_count = 0
pp = pprint.PrettyPrinter(indent=4)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if small_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if (frame_count == 0):
        # Find all the faces and face encodings in the current frame of video
        face_landmarks = face_recognition.face_landmarks(rgb_frame)

    frame_count = (frame_count + 1) % process_nth_frame

#    pp.pprint(face_landmarks)

    # Display the results
    for face in face_landmarks:
        if small_frame:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        
        for landmark, points in face.items():
            np_points = np.array(points, dtype='int32')
            cv2.polylines(frame, [np_points], False, (0,255,255), 3)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

