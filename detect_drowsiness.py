from toolkit.utils import Config
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def euclidean_dist(pointA, pointB):
	# compute the euclidean distance between the two points
	return np.linalg.norm(pointA - pointB)

def eye_aspect_ratio(eye):
	# euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	a = euclidean_dist(eye[1], eye[5])
	b = euclidean_dist(eye[2], eye[4])

	# euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	c = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (a + b) / (2.0 * c)

	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
	# euclidean distances between the three
	# vertical mouth landmarks (x, y)-coordinates
	a = euclidean_dist(mouth[1], mouth[7])
	b = euclidean_dist(mouth[2], mouth[6])
	c = euclidean_dist(mouth[3], mouth[5])

	# euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	d = euclidean_dist(mouth[0], mouth[4])

	# compute the mouth aspect ratio
	mar = (a + b + c) / (2.0 * d)

	# return the mouth aspect ratio
	return mar

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
	help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Config(args["conf"])

# if we are using GPIO/TrafficHat as an alarm
if conf["alarm"]:
	from gpiozero import TrafficHat
	th = TrafficHat()
	print("Using TrafficHat alarm...")

# initialize the frame center coordinates
centerX = None
centerY = None

# initialize a blink counter 
# yawn counter
# a boolean used to indicate if the alarm is going off
# start time
blinkCounter = 0
yawnCounter = 0
alarmOn = False
startTime = None

# load OpenCV's Haar cascade for face detection
# create the facial landmark predictor
print("Loading facial landmark predictor...")
detector = cv2.CascadeClassifier(conf["cascade_path"])
predictor = dlib.shape_predictor(conf["shape_predictor_path"])

# grab the indexes of the facial landmarks for the left, right eye,
# and inner part of the mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("Starting video stream thread...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream
	# resize
	# flip horizontally
	# convert to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# set the frame center
	if centerX is None and centerY is None:
		(centerX, centerY) = (frame.shape[1] // 2, frame.shape[0] // 2)

	# detect faces in the frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the detected faces
	for rect in rects:
		# draw a bounding box surrounding the face
		(x, y, w, h) = rect
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	# check if the number of faces detected is greater than zero
	if len(rects) > 0:
		# grab the face that's closest to the center
		centerRect = sorted(rects, key=lambda r: abs((
			r[0] + (r[2] / 2)) - centerX) + abs((
			r[1] + (r[3] / 2)) - centerY))[0]

		# get the coordinates of the rectangle in the center and
		# construct a dlib rectangle object from the Haar cascade
		# bounding box
		(x, y, w, h) = centerRect
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		# determine the facial landmarks for the face region
		# convert the facial landmark (x, y)-coordinates to np-array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		# this is mainly for debugging purposes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# if the ear is below the blink threshold
		if ear < conf["EYE_AR_THRESH"]:
			# increment the blink frame counter
			blinkCounter += 1

			# if the eyes were closed longer than config allows
			# sound the alarm
			if blinkCounter >= conf["EYE_AR_CONSEC_FRAMES"]:
				# if the alarm is not on, turn it on
				if not alarmOn:
					alarmOn = True

					# if trafficHAT is connected
					if conf["alarm"]:
						th.buzzer.blink(0.1, 0.1, 30,
							background=True)
						th.lights.red.blink(0.1, 0.1, 30,
							background=True)

				# draw an alarm on the frame
				cv2.putText(frame, "BEEP BEEP!", (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		# ear is not below the blink
		# reset the counter and alarm
		else:
			blinkCounter = 0
			alarmOn = False

		# extract the inner mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio for the mouth
		mouth = shape[mStart:mEnd]
		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

		# if the mar is above the yawn threshold
		if mar > conf["MOUTH_AR_THRESH"]:
			# increment the yawn frame counter
			# set the start time
			yawnCounter += 1
			startTime = datetime.now() if startTime == None else \
				startTime

			# if yawn frame counter is greater than config allows
			# and difference between current time
			# and start time is less than or equal to yawn threshold time
			if yawnCounter >= conf["YAWN_THRESH_COUNT"] and \
				(datetime.now() - startTime).seconds <= \
				conf["YAWN_THRESH_TIME"]:
				# if the alarm is not on, turn it on
				if not alarmOn:
					alarmOn = True

					# if TrafficHat 
					if conf["alarm"]:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)
						th.lights.red.blink(0.1, 0.1, 30,
							background=True)

				# draw an alarm on the frame
				cv2.putText(frame, "BEEP BEEP! YAWN!",
					(10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					(0, 0, 255), 2)

		# check to see if the start time is set
		elif startTime != None:
			# check if the difference between current time and start
			# time is greater than yawn threshold time
			if (datetime.now() - startTime).seconds > \
				conf["YAWN_THRESH_TIME"]:
				# reset yawn counter, alarm flag and start time
				yawnCounter = 0
				alarmOn = False
				startTime = None

		# draw the computed aspect ratios on the frame
		cv2.putText(frame, "EAR: {:.3f} MAR: {:.3f}".format(
			ear, mar), (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
			(0, 0, 255), 2)

	# if display flag is set, display the current frame
	# record if a user presses a key
	if conf["display"]:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# check to see if we have any open windows, and if so, close them
if conf["display"]:
	cv2.destroyAllWindows()

# release vs pointer
vs.stop()