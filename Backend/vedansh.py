import cv2
import requests
from flask import Flask, Response

app = Flask(__name__)

# Load video capture (0 for webcam or path to video file)
cap = cv2.VideoCapture(0)  # Use a video file like 'video.mp4' or a camera (0 or 1)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send the frame to the Super Resolution server (Server B)
            try:
                files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
                response = requests.post('http://192.168.86.115:5002/upload', files=files)
                print("Response from Server B:", response.json())
            except Exception as e:
                print(f"Error sending frame: {e}")

            # Yield the frame to stream it in the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on localhost at port 5000
