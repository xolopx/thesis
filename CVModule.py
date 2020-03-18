import TrackableObject
import CentroidTracker
import cv2 as cv
import numpy as np
import time







def define_contours(fgMask):

    contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_NONE)						# Look for contours in the foreground mask.

    threshedConts = []												    # Instantiate an empty list that will hold contours that meet the threshold

    # for i in range(len(contours)):			 						    # Delete contours that have a smallest area than the threshold.
    #     if cv.contourArea(contours[i]) > self.areaThresh:
    #         threshedConts.append(contours[i])

    # contours = threshedConts		 								    # Replaced list of contours with only those that passed thresh.

    contours_poly = [None] * len(contours)		 					    # Create list to hold contour polys.
    boundRect = []		 						    # Create list to hold contour best rectangle fits.

    for i, c in enumerate(contours):			 					    # Move through contours list generating enumerated pairs (indice, value).
        # contours_poly[i] = cv.approxPolyDP(c, 3, True)	 			    # Approximate a polyform contrackObjur +/- 3

        (x, y, w, h) = cv.boundingRect(c)	 		    # Generate bounding rect from the polyform contrackObjur. Returns "Upright Rectangle", i.e. Axis-aligned on bottrackObjm edge and whos eleft edge is vertical.
        if w >= 20 and h >= 25:                                       # Thresh bounding box by width and height.
            # boundRect[i] = cv.boundingRect(contours_poly[i])	 		# Generate bounding rect from the polyform contrackObjur. Returns "Upright Rectangle", i.e. Axis-aligned on bottrackObjm edge and whos eleft edge is vertical.
            boundRect.append(cv.boundingRect(c)) 		# Generate bounding rect from the polyform contrackObjur. Returns "Upright Rectangle", i.e. Axis-aligned on bottrackObjm edge and whos eleft edge is vertical.

    return boundRect                                                    # Return the bounding rectangles.


class CVModule:
    """
    Handles the detection, tracking, counting and speed estimation of an object given a
    series of images.
    """
    def __init__(self, inputVideo):
        """
        :param inputVideo: Video input to the module.
        """
        self.cenTrack = CentroidTracker.CentroidTracker()           # Object containing centroids detected for a given frame.
        self.objTracks = {}                                         # Dictionary of TrackbleObjects.
        self.frameCount = 0                                         # Number of frames of video processed.
        self.video = inputVideo                                     # Video from which to extract information.
        self.subtractor = cv.createBackgroundSubtractorMOG2(
            history=500, detectShadows=True)                        # Subtractor for procuring the input video's foreground objs.
        self.width = self.video.get(cv.CAP_PROP_FRAME_WIDTH)        # Width of input video
        self.height = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)      # Height of input video
        self.areaThresh = self.height*self.width/500                # Minimum area a contour must have to count as an object.
        self.countUp = 0                                            # Number of objects that have moved upward
        self.countDown = 0                                          # Number of objects that have moved downward
        self.struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)) 			# General purpose kernel.
        self.totalFrames = self.video.get(cv.CAP_PROP_FRAME_COUNT)

    def filter_frame(self,fgMask):
        """
        Applys morphology and median filtering to subtracted image to consolidate foreground objects
        and remove salt& pepper noise.
        :param fgMask: The foreground mask after applying subtractor
        """
    
        fgMask[fgMask < 240] = 0                                            # Threshold out shadows. (They're darker colored than pure foreground).
        fgMask = cv.medianBlur(fgMask,5)                                    # Apply median blur filter trackObj remove salt and pepper noise.
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, self.struct)           # Apply a closing trackObj join trackObjgether the surviving foreground blobs.
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, self.struct)                # Apply an opening trackObj remove some of the smaller and disconnected foreground blobs.
        fgMask = cv.dilate(fgMask, self.struct, iterations=2)                    # Apply dilation trackObj bolden the foreground objects.

        return fgMask


    def train_subtractor(self, trainNum=500):
        """
        Trains subtractor on the first N frames of video so it has a better idea
        of what the background consists of.
        :param trainNum: Number of training frames to be used on the model.
        """

        i = 0
        while i < trainNum:
            _, frame = self.video.read()
            self.subtractor.apply(frame, None, 0.001)
            i += 1

    def draw_info(self, image, boxes):
        """
        Marks frame count, up & down count, object speed, centroid and bounding boxes on a given image.
        :param image: Image to be drawn on.
        :param boxes: Bounding box information.
        """

        # Draw on the bounding boxes.
        for i in range(len(boxes)):
            cv.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])),
                (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])), (0, 255, 238), 2)

        # Draw centroids.
        for (objectID, centroid) in self.cenTrack.centroids.items():
            text = "ID {}".format(objectID)
            cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.circle(image, (centroid[0], centroid[1]), 4, (0,355, 0),-1)

        # Draw lines
        # cv.line(image, (0, 340), (640, 340), (255, 1, 255), 2)
        cv.line(image, (0, 250), (640, 250), (255, 1, 255), 2)

        # Draw on counts
        textUp = "Up {}".format(self.countUp)
        textDown = "Down {}".format(self.countDown)
        cv.putText(image, textUp, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv.putText(image, textDown, (500, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw speeds
        for (trackID, track) in self.objTracks.items():
            if not track.finished:											# Show speed until object's centroid is deregistered.
                center = track.centroids[-1]								# Get the last centroid in items history.
                x = center[0]
                y = center[1]
                textSpeed = "{:4.2f}".format(track.speed)
                # cv.putText(image, textSpeed, (x-10,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (20, 112, 250), 2)

        # Draw the number of frames
        cv.putText(image, "Frame: {}".format(self.frameCount), (280, 30),cv.FONT_HERSHEY_SIMPLEX, 0.5, (224, 9, 52, 2))
        self.draw_grid(image)


    def draw_grid(self,image):

        across = 0
        up = 0
        for i in range(image.shape[0] % 50):
            cv.line(image,(0,across),(int(self.width),across), (66, 135, 245), 1)
            across += 50

        for i in range(image.shape[1] % 50):
            cv.line(image,(up,0),(up,int(self.height)), (66, 135, 245), 1)
            up += 50



    def update_tracks(self):
        """ Generates and updates trackable objects with centroids data. """
        centroids = self.cenTrack.centroids                                 # Get the dicitonary of centroids out of the centroid tracker

        for (objectID, centroid) in centroids.items():			            # Loop through all centroids.

            trackObj = self.objTracks.get(objectID, None)		            # Check if a trackable object exists given a centroid's ID.

            if trackObj is None:								            # If there's no existing trackabe object for that centroid create a new one.
                trackObj = TrackableObject.TrackableObject(
                    objectID, centroid, self.frameCount)  	                # Create new trackable object to match centroid.

            else:												            # If the object does exists determine the direction it's travelling.
                y = [c[1] for c in trackObj.centroids]			            # Look at difference between y-coord of current centroid and mean of previous centroids.
                direction = centroid[1] - np.mean(y)			            # Get the difference.
                trackObj.centroids.append(centroid)		;		            # Assign the current centroid to the trackable objects history of centroids.
                thresh = 250                                                # Thresh value reflect position of the counting line.
                if not trackObj.counted : 						            # If the object hasn't been counted
                    if direction < 0 and centroid[1] < thresh:	            # If direction is up.
                        self.countUp += 1
                        trackObj.counted = True					            # Set obj as counted

                    elif direction > 0 and centroid[1] > thresh:            # Direction is down.
                        self.countDown += 1
                        trackObj.counted = True

            self.objTracks[objectID] = trackObj				                # Add the new trackable object trackObj the dictionary using its id as key.

        for trackID, track in self.objTracks.items():
            for ID in self.cenTrack.deregisteredID:
                if track.objectID == ID:
                    track.finished = True
                else:
                    track.currentFrame = self.frameCount



    def process(self):
        """
        Executes processing on video input. Resposible for:
         -
         -
         -
        :return:
        """

        self.train_subtractor()         # Initially, train the subtractor.


        """ MAIN LOOP """
        timeStart = time.time()
        while True and self.frameCount < (self.totalFrames-600):															# Loop will execute until all input processed or user exits.

            _, frame = self.video.read()									# Read out a frame of the input video.
            mask = self.subtractor.apply(frame)							    # Apply the subtractor trackObj the frame of the image trackObj get the foreground.

            mask = self.filter_frame(mask)                                       # Apply morphology, threshing and median filter.

            boundingRect = define_contours(mask)                       # Get the bounding boxes by locating contours in the foreground mask.

            objects, deregID = self.cenTrack.update(boundingRect)			# Update centroids by looking at the newest bounding rectangle information.

            self.update_tracks()						                    # Update trackable object statuses and statistics (up & down count).

            mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)			 			# Convert foreground mask to a 3-channel image.
            self.draw_info(mask, boundingRect)	 	 	 			        # Draw graphics onto mask
            self.draw_info(frame, boundingRect) 			 		        # Draw graphics onto frame

            combined = np.hstack((frame, mask))								# Stitch together original image and foreground mask for display.
            cv.imshow("Original", combined)	 								# Show the result.

            self.frameCount += 1											# Increment the number of frames.

            if self.frameCount % 1 == 0:                                    # To reduce frequency of determing object speed.
                for objID, objs in self.objTracks.items():
                    objs.calc_speed()
            key = cv.waitKey(10)
            if key == 27:
                break
            if key == ord('n'):
                while True:
                    key = cv.waitKey(50)
                    if key == ord('n'):
                        break
        print("Time Elapsed: {}".format(time.time()-timeStart))
        print("Frames Consumed: {}".format(self.frameCount))


