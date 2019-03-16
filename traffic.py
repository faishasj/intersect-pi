import cv2 as cv

# Value for Gaussian blur
BLUR_VALUE = 21

# Y value of line at which cars will be counted when crossed
COUNT_LINE = -75

# Error margin in pixels for cars crossing the COUNT_LINE
COUNT_ERROR_MARGIN = 2

# Threshold brightness value for blob detection
THRESHOLD_VALUE = 30

# Weight of the input image into the background accumulator frame
AVG_WEIGHT = 0.01


def main():
    # Set up the video capture
    cap = cv.VideoCapture("feed/videoplayback.mp4")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Read in the first frame to set up a background accumulator frame
    ret, frame = cap.read()
    avg_frame = frame.astype("float")

    no_of_cars = 0

    while ret:
        # Read in the current frame
        ret, frame = cap.read()

        # Accumulate an average of the frames to form the background frame
        cv.accumulateWeighted(frame, avg_frame, AVG_WEIGHT)

        # Convert current frame and background frame to grayscale and blur to remove noise
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (BLUR_VALUE, BLUR_VALUE), 0)

        avg_frame_res = cv.convertScaleAbs(avg_frame)
        gray_avg_frame = cv.cvtColor(avg_frame_res, cv.COLOR_BGR2GRAY)
        gray_avg_frame = cv.GaussianBlur(gray_avg_frame, (BLUR_VALUE, BLUR_VALUE), 0)

        # Subtract the current frame from the background frame to detect car blobs
        sub = cv.absdiff(gray_frame, gray_avg_frame)
        retval, thresh = cv.threshold(sub, THRESHOLD_VALUE, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Do more image processing to close up the contour and join adjacent blobs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.dilate(thresh, kernel, iterations=2)

        # Detect the car blobs
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Count the cars in each lane
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if x < frame_width / 2:
                if abs(y - (frame_height + COUNT_LINE)) < COUNT_ERROR_MARGIN:
                    no_of_cars += 1
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.line(frame, (0, frame_height + COUNT_LINE), (frame_width, frame_height + COUNT_LINE), (0, 0, 255), 2)
        cv.putText(frame, "NUMBER OF CARS: " + str(no_of_cars), (10, frame_height - 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Display the processed frames
        cv.imshow('Background', avg_frame_res)
        cv.imshow('Subtraction', sub)
        cv.imshow('Threshold', thresh)
        cv.imshow('Video', frame)

        # Press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()