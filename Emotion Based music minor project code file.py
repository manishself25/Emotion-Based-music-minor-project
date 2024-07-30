import cv2
import time
import numpy as np
from keras.models import load_model
from googleapiclient.discovery import build
from selenium import webdriver


# Load the Haar Cascade face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion recognition model
emotion_model = load_model(r"C:\Users\Manish Sharma\Desktop\minor\fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open a connection to the webcam (usually the default camera, index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Initialize a variable to track time
start_time = time.time()

# Initialize a list to store detected emotions
detected_emotions = []

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was read successfully
    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if 1 second has passed
        current_time = time.time()
        if current_time - start_time >= 1:
            # Capture an image and save it without the rectangle
            image_filename = f"captured_image_{int(current_time)}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Image saved as {image_filename}")
            start_time = current_time  # Reset the timer

        # For each detected face
        for (x, y, w, h) in faces:
            # Crop the face region from the frame
            face_roi = gray[y:y + h, x:x + w]

            # Resize the face region to match the input size of the emotion model (48x48)
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)

            # Perform emotion recognition on the face_roi
            emotion_predictions = emotion_model.predict(face_roi)
            emotion_label = emotion_labels[np.argmax(emotion_predictions)]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the predicted emotion label on the frame
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add the detected emotion to the list
            detected_emotions.append(emotion_label)

        cv2.imshow("Webcam", frame)

    # Check for user input
    key = cv2.waitKey(1)

    # Break the loop if the 'q' key is pressed
    if key & 0xFF == ord('q'):
        break

# Print the list of detected emotions
print("Detected Emotions:", detected_emotions)

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Set up YouTube Data API
api_key = 'AIzaSyC_ZRCsiw-2iPtTET5F_j6LJWEB7tkCQ9s'
youtube = build('youtube', 'v3', developerKey=api_key)

# Get user's mood input
print('Last Detected Emotion :',detected_emotions[-1])
# Search for music videos based on recommendations
search_response = youtube.search().list(
    type="video",
    part="id",
    maxResults=1  # Only get the first result
).execute()

if 'items' in search_response:
    for item in search_response['items']:
        if 'videoId' in item.get('id', {}):
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Video URL: {video_url}")

            # Use Selenium to open the URL in a browser and play the video
            driver = webdriver.Chrome()

            try:
                driver.get(video_url)
                # Wait for the video to load (you can adjust the time as needed)
                time.sleep(40)

                # Find and click the play button
                play_button = driver.find_element_by_css_selector(".ytp-play-button")
                if play_button:
                    play_button.click()
                    print("Clicked the play button.")
                else:
                    print("Play button not found.")

                # Wait for the video to play (adjust the time as needed)
                time.sleep(30)

            except Exception as e:
                print(f"Error: {e}")

            finally:
                driver.quit()  # Close the browser window when done

        else:
            print("VideoId not found in the item.")

            print(search_response)

else:
    print("No items found in the API response.")