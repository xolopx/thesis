import CVModule
import cv2 as cv


inputVideo = cv.VideoCapture("traffic_short.mp4")
process = CVModule.CVModule(inputVideo)
process.process()



