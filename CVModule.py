import TrackableObject
import CentroidTracker
import cv2 as cv


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
        self.objTracks = {}                                         # Dctionary of objects being tracked.
        self.frameCount = 0                                         # Number of frames of video processed.
        self.video = inputVideo                                     # Video from which to extract information.
        self.subtractor = cv.createBackgroundSubtractorMOG2(
            history=500, detectShadows=True)                        # Subtractor for procuring the input video's foreground objs.
        self.width = self.video.get(cv.CV_CAP_PROP_FRAME_WIDTH)     # Width of input video
        self.height = self.video.get(cv.CV_CAP_PROP_FRAME_HEIGHT)   # Height of input video


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

    def process(self):
        """
        Executes processing on video input. Resposible for:
         -
         -
         -
        :return:
        """

        self.train_subtractor()         # Initially, train the subtractor.
