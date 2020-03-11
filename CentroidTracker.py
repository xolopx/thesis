from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """ This is the constructor."""
    def __init__(self, maxDisappeared=100, maxDistance=100):
        self.nextObjectID = 0                       # Counter for object IDs
        self.objects = OrderedDict()                # Current centroids. Key: Unique ID, Value: Centroid (x,y)
        self.disappeared = OrderedDict()            # Centroids that are missing. Key: Unique ID, Value: # frames centroid has disappeared for.
        self.maxDisappeared = maxDisappeared        # Number of frames a centroid can go missing for before being removed.
        self.maxDistance = maxDistance              # The maximum distance a centroid can reappear from it's previous and still be associated.
    """ Registers a new centroid to be tracked."""
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid  # Next available unique ID is used for new centroid index in object list.
        self.disappeared[self.nextObjectID] = 0     # Initially the number of times the centroid has disappeared is 0.
        self.nextObjectID += 1                      # Increment the object ID counter.
    """ Reregisters a centroid from the tracked list."""
    def deregister(self, objectID):
        del self.objects[objectID]                  # Use objecID to index centroid and remove from list.
        del self.disappeared[objectID]              # Use objecID to index centroid and remove from list.
    """ Checks bounding box state against centroids state. 
        @Params:
            rects - list of bounding boxes (startX, startY, endX, endY)
            
        @Returns:
            Updated list of centroids. 
    """
    def update(self, rects):
        deregisteredID = []                                             # List of deregistered centroids. Will be refreshed for every frame.

        if len(rects) == 0:                                             # Check to see if the list of bounding boxes is empty.
            for objectID in list(self.disappeared.keys()):              # Loop over existing tracked objects and mark them as dissappered.
                self.disappeared[objectID] += 1                         # Increment all centroids lost count by 1.
                if self.disappeared[objectID] > self.maxDisappeared:    # Check if centroid has been missing too many frames.
                    self.deregister(objectID)                           # Deregister centroid.
                    deregisteredID.append(objectID)                     # Add to list of deregistered centroids.
            return self.objects, deregisteredID                         # Return early because there's no new centroids to check.

        inputCentroids = np.zeros((len(rects), 2), dtype='int')         # Initialize an array of input centroids for current frame.

        for (i, (startX, startY, width, height)) in enumerate(rects):   # Derive the centroid of the bounding box.
            cX = int((startX + (startX + width)) / 2.0)
            cY = int((startY + (startY + height)) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:                                      # If the number of existing centroids is zero.
            # 1. Get distance between all new centroids.
            # 2. If their are any centroids that are within threshold of closeness, combine them into one entity.
                # 2a. Determine which centroid to use as the base, somehow.
                # 2b. Remove the discarded centroid from input list.

            for i in range(0, len(inputCentroids)):                     # Register all centroids as new.
                self.register(inputCentroids[i])

        else:                                                           # Else match new centroids with existing ones.
            objectIDs = list(self.objects.keys())                       # Extract existing centroid IDs.
            objectCentroids = list(self.objects.values())               # Extract centroid co-ords.


            D = dist.cdist(np.array(objectCentroids), inputCentroids)   # D is an array of shape (# new centroids, # existing centroids) of distances. Cells are distances between centroids.
            rows = D.min(axis=1).argsort()                              # Return list of inidices representing rows in order of rows with the smallest values.
            cols = D.argmin(axis=1)[rows]                               # Return index of column with mimimum value for each row.
                                                                        # With the rows and cols indice lists we have the smallest distance value's
                                                                        # coordinates for each existing centroid from the D array.
                                                                        # We sort the list of pairs by smallest, because we want to absolute closest pairs to be consumed first, otherwise
                                                                        # Another centroid might consume "its" closest over some other centroid who is even closer.

            usedRows = set()                                            # Tracks which pair have already been used.
            usedCols = set()

            for (row, col) in zip(rows, cols):                          # Loop over the minimum distance pairs.
                if row in usedRows or col in usedCols:                  # Skip an existing or new centroid if it has already been used.
                    continue

                """ DEBUG """
                objectID = objectIDs[row]                               # Get old centroid's objectID.

                distancio = D[row][col]                                 # DEBUG: get distance between pair.
                print("Centroid {0} found {1:5.2f} away".format(objectID, distancio))


                if D[row][
                    col] < self.maxDistance:                      # if the distance of the nearest centroid is satisfies distance thresh.
                    self.objects[objectID] = inputCentroids[col]        # Set the old centroid's new location to be the closest new centroid.
                    self.disappeared[objectID] = 0                      # Reset the centroid's disappeared value as its location has been updated.
                else:
                    self.deregister(objectID)                           # If the centoid is too far away deregister.
                    deregisteredID.append(objectID)                     # Append the deregistered ID to list to give to trackers, to mark tracker as deregistered.
                # Set the used row and col so they aren't assigned again.
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows) # Find existing centroids that weren't used.
            unusedCols = set(range(0, D.shape[0])).difference(usedCols) # Find new centroids that weren't used.

            if D.shape[0] >= D.shape[1]:                                # If the # of old centroids >= # of new ones.
                for row in unusedRows:                                  # Loop over unused rows.
                    objectID = objectIDs[row]                           # Get the object ID of the unused existing centroid.
                    self.disappeared[objectID] += 1                     # Count that the centroid has been missing for a frame.

                    if self.disappeared[objectID] > self.maxDisappeared:# If missing for > threshold
                        self.deregister(objectID)                       # Deregister centroid
                        deregisteredID.append(objectID)                 # Add deregistered centroid ID to the list.
            else:
                for col in unusedCols:                                  # Check for unused new centroids
                    # 1. Get distance between unused new centroids and existing ones.
                    # 2. If any new centroids are within threshold distance of existing, then associate them.
                        # 2a. Use existing centroid as the base
                        # 2b. Remove the assimilated centroid from input list.
                    self.register(inputCentroids[col])                  # Register any new centroids.

        return self.objects, deregisteredID                             # Return the set of updated centroids.











