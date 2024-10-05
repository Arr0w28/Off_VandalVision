# Server A: Stream video over HTTP

from flask import Flask, Response
import cv2

app = Flask(__name__)

# Load video capture (0 for webcam or path to video file)
cap = cv2.VideoCapture(0)  # Use a video file like 'video.mp4' or a camera (0 or 1)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame as byte chunks over HTTP
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running on localhost at port 5000
