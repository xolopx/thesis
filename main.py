import CVModule
import cv2 as cv


inputVideo = cv.VideoCapture("/home/tom/Desktop/pycharm_projects/tracker/traffic.mp4")
process = CVModule.CVModule(inputVideo)
process.process()



