import cv2 as cv

# --------------------------------------------------------------------------------------------
# car detection

# # input video
# handle_read = cv.VideoCapture('./slow_traffic_small.mp4')
# frame_width = int(handle_read.get(cv.CAP_PROP_FRAME_WIDTH))
# frame_height = int(handle_read.get(cv.CAP_PROP_FRAME_HEIGHT))
# four_cc = int(handle_read.get(cv.CAP_PROP_FOURCC))
# fps = int(handle_read.get(cv.CAP_PROP_FPS))


# # Define the half resolution frame size
# half_res = (frame_width // 2, frame_height // 2)

# # output video
# handle_write = cv.VideoWriter(
#     'detectedcars.mp4', four_cc, fps, half_res)

# # cascade classifier
# car_classifier = cv.CascadeClassifier()
# if not car_classifier.load('./cars.xml'):
#     print('Error loading cars cascade')
#     exit(0)

# frame_number = 0

# while handle_read.isOpened():
#     print(frame_number)
#     ret, frame = handle_read.read()
#     if ret is True:
#         frame_number += 1

#         # Resize the frame to half resolution
#         frame = cv.resize(frame, half_res)
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         cars = car_classifier.detectMultiScale(gray, 1.4, 2)

#         for (x, y, w, h) in cars:
#             frame = cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h),
#                                  color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)
#         handle_write.write(frame)
#     else:
#         break

# handle_read.release()
# handle_write.release()

# print(f'{frame_number} frames read and processed')
# --------------------------------------------------------------------------------------------
# face detection haarcascade


# # input video
# handle_read = cv.VideoCapture('./people.mp4')
# frame_width = int(handle_read.get(cv.CAP_PROP_FRAME_WIDTH))
# frame_height = int(handle_read.get(cv.CAP_PROP_FRAME_HEIGHT))
# four_cc = int(handle_read.get(cv.CAP_PROP_FOURCC))
# fps = int(handle_read.get(cv.CAP_PROP_FPS))

# # Define the quarter resolution frame size
# quarter_resolution = (frame_width // 4, frame_height // 4)

# # output video
# handle_write = cv.VideoWriter(
#     'peoplefaces_lowreshaar.mp4', four_cc, fps, quarter_resolution)

# # cascade classifier
# classifier = cv.CascadeClassifier()
# if not classifier.load('./haarcascade_profileface.xml'):
#     print('Error loading face cascade')
#     exit(0)

# frame_number = 0

# while handle_read.isOpened():
#     print(frame_number)
#     ret, frame = handle_read.read()
#     if ret is True:
#         frame_number += 1

#         # Resize the frame to quarter resolution
#         frame = cv.resize(frame, quarter_resolution)

#         faces = classifier.detectMultiScale(
#             frame, 1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
#         # //argumentos custom
#         # faces = classifier.detectMultiScale(frame) //args original

#         for (x, y, w, h) in faces:
#             frame = cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h),
#                                  color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)
#         handle_write.write(frame)
#     else:
#         break

# handle_read.release()
# handle_write.release()

# print(f'{frame_number} frames read and processed')
# --------------------------------------------------------------------------------------------
# # face detection lbpcascade

# input video
handle_read = cv.VideoCapture('./people.mp4')
frame_width = int(handle_read.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(handle_read.get(cv.CAP_PROP_FRAME_HEIGHT))
four_cc = int(handle_read.get(cv.CAP_PROP_FOURCC))
fps = int(handle_read.get(cv.CAP_PROP_FPS))

# Define the quarter resolution frame size
quarter_resolution = (frame_width // 4, frame_height // 4)

# output video
handle_write = cv.VideoWriter(
    'peoplefaces_lowreslbp.mp4', four_cc, fps, quarter_resolution)

# cascade classifier
classifier = cv.CascadeClassifier()
if not classifier.load('./lbpcascade_profileface.xml'):
    print('Error loading face cascade')
    exit(0)

frame_number = 0

while handle_read.isOpened():
    print(frame_number)
    ret, frame = handle_read.read()
    if ret is True:
        frame_number += 1

        # Resize the frame to quarter resolution
        frame = cv.resize(frame, quarter_resolution)

        faces = classifier.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            frame = cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h),
                                 color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)
        handle_write.write(frame)
    else:
        break

handle_read.release()
handle_write.release()

print(f'{frame_number} frames read and processed')
# --------------------------------------------------------------------------------------------
