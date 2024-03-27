import pafy
import cv2

# url = "http://192.168.0.103"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# Create a VideoCapture object.
cap = cv2.VideoCapture("http://192.168.0.103:81/stream")

# Check if the video was opened successfully.
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Read the video frames.
while True:
    # Capture a frame.
    ret, frame = cap.read()

    # If the frame is empty, break the loop.
    if not ret:
        break

    # Display the frame.
    cv2.imshow("Video", frame)

    # Wait for a key press.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object.
cap.release()

# Close all windows.
cv2.destroyAllWindows()