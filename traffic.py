import cv2 as cv
import numpy as np
import pandas as pd
from database import Database
from sklearn.cluster import KMeans

# Weight of the input image into the background accumulator frame
AVG_WEIGHT = 0.01

# Kernel for Gaussian blur
BLUR_KERNEL = (5, 5)

# Y value of line at which cars will be counted when crossed
COUNT_LINE = -75

# Error margin in pixels for cars crossing the COUNT_LINE
COUNT_ERROR_MARGIN = 2

# Number of midpoint samples to take before determining lanes
MAX_MIDPOINT_SAMPLE = 50

# Size of the structuring element for the kernel
STRUCT_SIZE = (10, 10)

# Threshold brightness value for blob detection
THRESHOLD_VALUE = 30

# Path to video for computer vision
FEED = "feed/feed_west.mp4"


def kmcluster(dataset, k):
    """ Find k number of means by analysing clusters in the dataset. """
    x = pd.DataFrame(np.array(dataset).reshape(len(dataset), 1), columns=list("x"))
    y = pd.DataFrame(np.array([1] * len(dataset)).reshape(len(dataset), 1), columns=list("y"))
    kmeans = KMeans(n_clusters=k).fit(x, y)
    klist = list(kmeans.cluster_centers_)
    clusters = []
    for i in range(len(klist)):
        clusters.append(int(klist[i][0]))
    return clusters


def min_index(a_list):
    """ Return the index of the item with lowest value. """
    min_val = a_list[0]
    min_i = 0
    for i in range(1, len(a_list)):
        if a_list[i] < min_val:
            min_val = a_list[i]
            min_i = i
    return min_i


def preprocess_frame(frame):
    """ Convert frame to grayscale and apply Gaussian blur to reduce noise. """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, BLUR_KERNEL, 0)
    return frame


def main():
    """ Process the traffic feed and update the database. """
    # Set up the video capture
    cap = cv.VideoCapture("feed/feed_west.mp4")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    db = Database()

    # Read in the first frame to set up a background accumulator frame
    ret, frame = cap.read()
    avg_frame = frame.astype("float")

    lane_count = [0, 0, 0]
    midpoints = []
    midpoint_clusters = []

    while ret:
        # Read in the current frame
        ret, frame = cap.read()

        # Accumulate an average of the frames to form the background frame
        cv.accumulateWeighted(frame, avg_frame, AVG_WEIGHT)
        avg_frame_res = cv.convertScaleAbs(avg_frame)

        # Convert current frame and background frame to grayscale and blur to remove noise
        gray_avg_frame = preprocess_frame(avg_frame_res)
        gray_frame = preprocess_frame(frame)

        # Subtract the current frame from the background frame to detect car blobs
        sub = cv.absdiff(gray_frame, gray_avg_frame)
        retval, thresh = cv.threshold(sub, THRESHOLD_VALUE, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Do more image processing to close up the contour and join adjacent blobs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, STRUCT_SIZE)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.dilate(thresh, kernel, iterations=2)

        # Detect the car blobs
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Count the cars in each lane
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if abs(y - (frame_height + COUNT_LINE)) < COUNT_ERROR_MARGIN and x < frame_width//2:
                midpoint = x + w / 2
                if len(midpoints) < MAX_MIDPOINT_SAMPLE:
                    midpoints.append(midpoint)
                else:
                    lane_diff = []
                    for i in midpoint_clusters:
                        lane_diff.append(abs(midpoint - i))
                    lane = min_index(lane_diff)
                    lane_count[lane] += 1
                    db.add_car(FEED[10], lane)

            #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(midpoints) == MAX_MIDPOINT_SAMPLE:
            midpoint_clusters = kmcluster(midpoints, 3)
            midpoints.append(0)

        if len(midpoints) > MAX_MIDPOINT_SAMPLE:
            for i in midpoint_clusters:
                cv.line(frame, (int(i),0),(int(i),frame_height),(0, 0, 255), 2)


        cv.line(frame, (0, frame_height + COUNT_LINE), (265, frame_height + COUNT_LINE), (0, 0, 255), 2)
        cv.putText(frame, str(lane_count[0]), (44, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(frame, str(lane_count[1]), (126, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(frame, str(lane_count[2]), (205, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Display the processed frames
        cv.imshow('Background', avg_frame_res)
        cv.imshow('Threshold', thresh)
        cv.imshow('Video', frame)

        # Press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()