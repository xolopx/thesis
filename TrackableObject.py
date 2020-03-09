class TrackableObject:
    def __init__(self,objectID, centroid, frameSt):
        self.objectID = objectID            # Unique identifier for the object
        self.centroids = [centroid]         # History of centroids stored in a list
        self.counted = False                # Marks if an object has been counted
        self.frameSt = frameSt              # Log the frame that the object entered.
        self.currentFrame = 0               # Latest value of frame.
        self.speed = 0                      # Speed of object
        self.finished = False               # Flag to mark if a trackable object has disappeared before passing threshold. (Because current solution is incomplete need to account for this happening).

    def calc_speed(self):
        """
        Calculates the speed of the object based on when it entered and exited frame.

        :return: The average speed of the object over this distance
        """
        length = len(self.centroids)

        y1 = self.centroids[0][1]                       # Get y1
        y2 = self.centroids[length-1][1]                # Get y2


        distance = abs(y1 - y2)                             # Get the first centroid and last centroid and update speed.
        if distance > 0:
            self.speed = (((self.currentFrame - self.frameSt)*0.033)/(distance*0.05))*3.6        # Number of pixels moved in a frame,
        return self.speed
