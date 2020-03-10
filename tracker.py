import cv2 as cv
import numpy as np
import CentroidTracker
import dlib
import TrackableObject


def draw_info(image, boxes, centroids, countUp, countDown, trackableObjects,frameCount):
	"""
	Marks important information on a given image.
	:param image: Image to be drawn on.
	:param boxes: Bounding box information.
	:param centroids: Centroid information.
	:return: Nil
	"""

	# Draw on the bounding boxes.
	for i in range(len(boxes)):
		cv.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])),
			(int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])), (0, 255, 238), 2)

	# Draw centroids.
	for (objectID, centroid) in centroids.items():
		text = "ID {}".format(objectID)
		cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv.circle(image, (centroid[0], centroid[1]), 4, (0,355, 0),-1)

	# Draw lines
	# cv.line(image, (0, 340), (640, 340), (255, 1, 255), 2)
	cv.line(image, (0, 250), (640, 250), (255, 1, 255), 2)

	# Draw on counts
	textUp = "Up {}".format(countUp)
	textDown = "Down {}".format(countDown)
	cv.putText(image, textUp, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
	cv.putText(image, textDown, (500, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

	# Draw speeds
	for (trackID, track) in trackableObjects.items():
		if not track.finished:											# Show speed until object's centroid is deregistered.
			center = track.centroids[-1]								# Get the last centroid in items history.
			x = center[0]
			y = center[1]
			textSpeed = "{:4.2f}".format(track.speed)
			# cv.putText(image, textSpeed, (x-10,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (20, 112, 250), 2)

	# Draw the number of frames
	cv.putText(image, "Frame: {}".format(frameCount), (280, 30),cv.FONT_HERSHEY_SIMPLEX, 0.5, (224, 9, 52, 2))


def update_tracks(objects, trackableObjects, h, countUp, countDown, frameCount, deregID):
	"""
	Generates and updates trackable objects with centroids data.

	:param deregID: IDs of deregistered centroids.
	:param frameCount: Count of frames processed.
	:param objects: Dictionary of centroids
	:param trackableObjects: Dictionary of trackable objects.
	:param h: Height of a frame of input.
	:param countUp: Counts objects moving up.
	:param countDown: Counts object moving down.
	:return: The updated dictionary of trackable objects. The count for up and down directions.
	"""

	for (objectID, centroid) in objects.items():			# Loop through all centroids.

		trackObj = trackableObjects.get(objectID, None)		# Check if a trackable object exists given a centroid's ID.

		if trackObj is None:								# If there's no existing trackabe object for that centroid create a new one.
			trackObj = TrackableObject.TrackableObject(
				objectID, centroid, frameCount)  			# Create new trackable object to match centroid.

		else:												# If the object does exists determine the direction it's travelling.
			y = [c[1] for c in trackObj.centroids]			# Look at difference between y-coord of current centroid and mean of previous centroids.
			direction = centroid[1] - np.mean(y)			# Get the difference.
			trackObj.centroids.append(centroid)				# Assign the current centroid to the trackable objects history of centroids.
			thresh = 250
			if not trackObj.counted : 						# If the object hasn't been counted
				if direction < 0 and centroid[1] < thresh:	# If direction is up.
					countUp += 1
					trackObj.counted = True					# Set obj as counted

				elif direction > 0 and centroid[1] > thresh:# Direction is down.
					countDown += 1
					trackObj.counted = True

		trackableObjects[objectID] = trackObj				# Add the new trackable object trackObj the dictionary using its id as key.

	for trackID, track in trackableObjects.items():
		for ID in deregID:
			if track.objectID == ID:
				track.finished = True
			else:
				track.currentFrame = frameCount
	return countUp, countDown


def train_bg_subtractor(subtractor, cap, num=500):
	"""

	:param subtractor: The subtractor instance.
	:param cap: The input video stream.
	:param num:	NUmber of frames to be used on the model.
	:return:
	"""

	i = 0
	while i < num:
		_, frame = cap.read()
		subtractor.apply(frame, None, 0.001)
		i += 1


def main():

	""" SETUP """
	subtractor = cv.createBackgroundSubtractorMOG2(history=500, detectShadows=True) 	# Keeping shadows and threshing them out later.
	try:
		capture = cv.VideoCapture("/home/tom/Desktop/pycharm_projects/tracker/traffic3.mp4")							# Open the input file.

	except FileNotFoundError:
		print("Could not find video")

	train_bg_subtractor(subtractor,capture,num=500)
	ct = CentroidTracker.CentroidTracker()								# Create the centroid tracking object.
	trackableObjects = {}												# Dictionary of objects being tracked.

	w = capture.get(3)													# Get the width of the frames.
	h = capture.get(4)													# Get the height of the frames
	areaThresh = h*w/500												# Approximate the smallest contour size in the ROI.
	kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3,3)) 		# Create a opening kernel
	kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (17,17)) 	# Create a closing kernel
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)) 			# General purpose kernel.

	frameCount = 0														# Count number of frames
	countUp = 0															# Count of objects moving up
	countDown = 0														# Count of objects moving down

	deregID = []														# List of IDs for deregistered centroids. Used to mark old trackable objects.

	""" MAIN LOOP """
	while True:															# Loop will execute until all input processed or user exits.

		_, frame = capture.read()										# Read out a frame of the input video.
		mask = subtractor.apply(frame)									# Apply the subtractor trackObj the frame of the image trackObj get the foreground.
		# ret, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)		# Threshold the foreground mask trackObj remove shadows.
		mask[mask < 240] = 0
		mask = cv.medianBlur(mask,5)									# Apply median blur filter trackObj remove salt and pepper noise.

		mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)				# Apply a closing trackObj join trackObjgether the surviving foreground blobs.
		mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)				# Apply an opening trackObj remove some of the smaller and disconnected foreground blobs.
		mask = cv.dilate(mask, kernel, iterations=2)					# Apply an opening trackObj remove some of the smaller and disconnected foreground blobs.

		contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,
									  cv.CHAIN_APPROX_NONE)										# Look for contours in the foreground mask. EXPLAIN PARAMETERS USED.

		threshedConts = []												# Instantiate an empty list that will hold contours that meet the threshold
		for i in range(len(contours)):			 						# Delete contours that have a smallest area than the threshold.
			if cv.contourArea(contours[i]) > areaThresh:
				threshedConts.append(contours[i])
		contours = threshedConts		 								# Make contours equal trackObj the updated list

		contours_poly = [None] * len(contours)		 					# Instantiate a list trackObj hold the poly form contours. It's length matches the number of contours found.
		boundRect = [None] * len(contours)		 						# Instantiate a list trackObj hold the bounding rect corner coordinates. Length matches the # contours.

		for i, c in enumerate(contours):			 					# Move through contours list generating enumerated pairs (indice, value).
			contours_poly[i] = cv.approxPolyDP(c, 3, True)	 			# Approximate a polyform contrackObjur +/- 3
			boundRect[i] = cv.boundingRect(contours_poly[i])	 		# Generate bounding rect from the polyform contrackObjur. Returns "Upright Rectangle", i.e. Axis-aligned on bottrackObjm edge and whos eleft edge is vertical.

		objects, deregID = ct.update(boundRect)							# Use the centroid tracking object trackObj update centroids.
		countUp, countDown = update_tracks(
			objects, trackableObjects, h, countUp,
			countDown, frameCount, deregID)								# Update trackable objects

		mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)			 			# Convert the binary output of the subtractor trackObj a 3 channel RGB image so it can be place in the same array as the original.
		draw_info(
			mask, boundRect, objects, countUp,
			countDown,trackableObjects, frameCount)	 		 			# Draw important graphics onto mask
		draw_info(
			frame, boundRect, objects, countUp,
			countDown, trackableObjects, frameCount) 					# Draw important graphics onto frame

		combined = np.hstack((frame, mask))								# Combine the original and mask intrackObj single image.
		cv.imshow("Original", combined)	 								# Show the result.

		frameCount += 1													# Increment the number of frames.

		if frameCount % 1 == 0:											# Update speed reading every 20 frames.
			for objID, objs in trackableObjects.items():
				objs.calc_speed()
		key = cv.waitKey(20)
		if key == 27:
			break
		if key == ord('n'):
			while True:
				key = cv.waitKey(20)
				if key == ord('n'):
					break

aNumber = 0
main()



