import cv2 as cv

# Weight of the input image into the background accumulator frame
AVG_WEIGHT = 0.01

# Kernel for Gaussian blur
BLUR_KERNEL = (9, 9)

# Y value of line at which cars will be counted when crossed
COUNT_LINE = -75

# Error margin in pixels for cars crossing the COUNT_LINE
COUNT_ERROR_MARGIN = 3

# Size of the structuring element for the kernel
STRUCT_SIZE = (6, 6)

# Threshold brightness value for blob detection
THRESHOLD_VALUE = 30


def preprocess_frame(frame):
    """ Convert frame to grayscale and apply Gaussian blur to reduce noise. """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, BLUR_KERNEL, 0)
    return frame


def main():
    # Set up the video capture
    cap = cv.VideoCapture("feed/videoplayback.mp4")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Read in the first frame to set up a background accumulator frame
    ret, frame = cap.read()
    avg_frame = frame.astype("float")

    lane_count = [0, 0, 0]
    lane_density = [0, 0, 0]

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
            if abs(y - (frame_height + COUNT_LINE)) < COUNT_ERROR_MARGIN:
                midpoint = x + w / 2
                if midpoint < 110:
                    lane_count[0] += 1
                elif midpoint < 186:
                    lane_count[1] += 1
                elif midpoint < 265:
                    lane_count[2] += 1

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.line(frame, (0, frame_height + COUNT_LINE), (265, frame_height + COUNT_LINE), (0, 0, 255), 2)
        cv.putText(frame, str(lane_count[0]), (44, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(frame, str(lane_count[1]), (126, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(frame, str(lane_count[2]), (205, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Display the processed frames
        cv.imshow('Background', avg_frame_res)
        cv.imshow('Threshold', thresh)
        cv.imshow('Video', frame)
        #cv.imshow('Lanes', canny)

        # Press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()