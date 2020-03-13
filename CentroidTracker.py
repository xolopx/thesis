from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

    def __init__(self, maxDisappeared=25, maxDistance=100, minDistance=30):
        """
        :param maxDisappeared: Maximum number of frame a centroid can disappear for before being deregistered.
        :param maxDistance: Maximum distance a centroid can reappear and still be associated with its nearest centroid.
        :param minDistance: Minimum distance a centroid can be to another without getting associated with that centroid.
        """
        self.deregisteredID = []                                             # List of deregistered centroids. Will be refreshed for every frame.
        self.nextObjectID = 0                       # Counter for object IDs
        self.centroids = OrderedDict()                # Current centroids. Key: Unique ID, Value: Centroid (x,y)
        self.disappeared = OrderedDict()            # Centroids that are missing. Key: Unique ID, Value: # frames centroid has disappeared for.
        self.maxDisappeared = maxDisappeared        # Number of frames a centroid can go missing for before being removed.
        self.maxDistance = maxDistance              # The maximum distance a centroid can reappear from it's previous and still be associated.
        self.minDistance = minDistance              # Min distance a centroid can appear near other centroid and not be associated with it.
        self.deregisteredID = []                    # list of IDs of centroids that have disappeared into to the nether.

    def register(self, centroid):
        """ Registers a new centroid to be tracked."""
        self.centroids[self.nextObjectID] = centroid  # Next available unique ID is used for new centroid index in object list.
        self.disappeared[self.nextObjectID] = 0     # Initially the number of times the centroid has disappeared is 0.
        self.nextObjectID += 1                      # Increment the object ID counter.

    def deregister(self, objectID):
        """ Deregisters a centroid with ID objectID."""
        del self.centroids[objectID]                  # Use objecID to index centroid and remove from list.
        del self.disappeared[objectID]              # Use objecID to index centroid and remove from list.

    def update(self, rects):
        """
        Checks bounding box state against centroids state.
        @Params:
            rects - list of bounding boxes (startX, startY, endX, endY)

        @Returns:
            Updated list of centroids.
        """

        if len(rects) == 0:                                             # Check to see if the list of bounding boxes is empty.
            for objectID in list(self.disappeared.keys()):              # Loop over existing tracked centroids and mark them as dissappered.
                self.disappeared[objectID] += 1                         # Increment all centroids lost count by 1.
                if self.disappeared[objectID] > self.maxDisappeared:    # Check if centroid has been missing too many frames.
                    self.deregister(objectID)                           # Deregister centroid.
                    self.deregisteredID.append(objectID)                     # Add to list of deregistered centroids.
            return self.centroids, self.deregisteredID                         # Return early because there's no new centroids to check.

        inputCentroids = np.zeros((len(rects), 2), dtype='int')         # Initialize an array of input centroids for current frame.

        for (i, (startX, startY, width, height)) in enumerate(rects):   # Derive the centroid of the bounding box.
            cX = int((startX + (startX + width)) / 2.0)
            cY = int((startY + (startY + height)) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.centroids) == 0:                                      # If the number of existing centroids is zero.

            inputCentroidsD = dist.cdist(
                np.array(inputCentroids), inputCentroids)               # Calculate distances between all new centroids.
            dim = np.shape(inputCentroids)                              # Get dimensions of distance matrix. NB:It's square.

            for row in range(dim[0]):
                for col in range(dim[0]):                               # Cycle through pairs.
                    if ((inputCentroidsD[row][col] < self.minDistance)  # If pair is sufficiently close together, combine.
                            and (row != col)):                          # Centroid to itself is 0, so ignore these scenarios.
                        print("Merged input centroids\n")
                        np.delete(inputCentroids, col, axis=0)          # Delete row for that centroid. Array is vertical list of points: (x,y).


            for i in range(0, len(inputCentroids)):                     # Register all centroids as new.
                self.register(inputCentroids[i])

        else:                                                           # Else match new centroids with existing ones.
            objectIDs = list(self.centroids.keys())                       # Extract existing centroid IDs.
            objectCentroids = list(self.centroids.values())               # Extract centroid co-ords.


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

                objectID = objectIDs[row]                               # Get old centroid's objectID.

                if D[row][col] < self.maxDistance:                      # If the distance of the nearest centroid is satisfies distance thresh.
                    self.centroids[objectID] = inputCentroids[col]        # Set the old centroid's new location to be the closest new centroid.
                    self.disappeared[objectID] = 0                      # Reset the centroid's disappeared value as its location has been updated.
                else:
                    self.deregister(objectID)                           # If the centoid is too far away deregister.
                    self.deregisteredID.append(objectID)                     # Append the deregistered ID to list to give to trackers, to mark tracker as deregistered.

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows) # Find existing centroids that weren't used.
            unusedCols = set(range(0, D.shape[1])).difference(usedCols) # Find new centroids that weren't used.

            if D.shape[0] >= D.shape[1]:                                # If the # of old centroids >= # of new ones.
                for row in unusedRows:                                  # Loop over unused rows.
                    objectID = objectIDs[row]                           # Get the object ID of the unused existing centroid.
                    self.disappeared[objectID] += 1                     # Count that the centroid has been missing for a frame.

                    if self.disappeared[objectID] > self.maxDisappeared:# If missing for > threshold
                        self.deregister(objectID)                       # Deregister centroid
                        self.deregisteredID.append(objectID)                 # Add deregistered centroid ID to the list.
            else:
                combined = []                                           # List of new centroid indices that are genuine new centroids.
                for col in unusedCols:                                  # Check for unused new centroids
                    for row in range(D.shape[0]):                       # Check unused centroids against all existing centroids.
                        if D[row][col] < self.minDistance:              # Check if distance is < minimum.
                            combined.append(col)                        # Append the index to the list of combined centroids.
                            # print("Centroid absorbed by {} at {}".format(objectIDs[row], objectCentroids[objectIDs[row]]))

                for col in unusedCols:
                    if col not in combined:                             # This centroid has not been absorbed into another.
                        self.register(inputCentroids[col])

        return self.centroids, self.deregisteredID                             # Return the set of updated centroids.











