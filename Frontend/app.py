import streamlit as st
import cv2
from sqlalchemy import create_engine, Column, String, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import deque
import datetime
import time
import os


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # 3 classes: spitting, graffiti, no graffiti

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = SimpleNet()  
model.load_state_dict(torch.load('/Users/vedanshkumar/Documents/GitHub/Off_VandalVision/Detection/classification_model.pth'))  # Load the state dict
model.eval()  # Set the model to evaluation mode

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to process the video frame and run model detection
def process_video_frame(frame):
    """Convert frame, run through the model, and return the detected label."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)
    img_tensor = transform(img_pil).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)  # Forward pass
        outputs = outputs.squeeze()
        outputs = outputs.flip(dims=[0])
        predicted = torch.argmax(torch.softmax(outputs,dim=0),dim=0)  # Get the predicted class

    # Return the label
    return predicted.item()




# Set page configuration
st.set_page_config(page_title="Security Monitor", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("/Users/vedanshkumar/Documents/GitHub/Off_VandalVision/Frontend/style.css")

# Database setup (same as before)
engine = create_engine('sqlite:///users.db')
meta = MetaData()

users = Table(
    'users', meta,
    Column('id', Integer, primary_key=True),
    Column('username', String, unique=True),
    Column('password', String)
)

meta.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# User management functions (same as before)
def add_user(username, password):
    try:
        insert = users.insert().values(username=username, password=password)
        session.execute(insert)
        session.commit()
        return True
    except IntegrityError:
        session.rollback()
        return False

def authenticate(username, password):
    query = users.select().where(users.c.username == username).where(users.c.password == password)
    result = session.execute(query).fetchone()
    return result is not None

# Updated sidebar for navigation
def sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: white;'>Navigation</h2>", unsafe_allow_html=True)
        if st.button("Home", key="nav_home"):
            st.session_state['page'] = 'home'
        if st.button("Live Feed", key="nav_live"):
            st.session_state['page'] = 'live'
        if st.button("Real-Time Analysis", key="nav_realtime"):
            st.session_state['page'] = 'realtime'
        if st.button("Detect Vandalism", key="nav_detect"):
            st.session_state['page'] = 'detect'
        if st.button("Logout", key="nav_logout"):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

# Login Page
def login():
    st.markdown("<h1 style='text-align: center; color: white;'>Vindhler</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", key="login"):
            if authenticate(username, password):
                st.success(f"Welcome, {username}!")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['page'] = 'home'
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Don't have an account?</p>", unsafe_allow_html=True)
        if st.button("Register", key="goto_register"):
            st.session_state['page'] = 'register'
            st.rerun()

# Register Page
def register():
    st.markdown("<h1 style='text-align: center; color: white;'>Create an Account</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        
        if st.button("Register", key="register"):
            if add_user(new_username, new_password):
                st.success("User registered successfully!")
                st.session_state['page'] = 'login'
                st.rerun()
            else:
                st.error("Username already exists. Please choose another.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Already have an account?</p>", unsafe_allow_html=True)
        if st.button("Back to Login", key="goto_login"):
            st.session_state['page'] = 'login'
            st.rerun()

# Home Page
def home():
    st.markdown("<h1 style='text-align: center; color: white;'>Security Monitor Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: grey;'>Welcome, {st.session_state['username']}!</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h3 style='color: grey;'>Live Feed</h3>", unsafe_allow_html=True)
        st.image("https://via.placeholder.com/300x200.png?text=Live+Feed+Preview", use_column_width=True)
        if st.button("Go to Live Feed"):
            st.session_state['page'] = 'live'
            st.rerun()
    
    with col2:
        st.markdown("<h3 style='color: grey;'>Real-Time Analysis</h3>", unsafe_allow_html=True)
        st.image("https://via.placeholder.com/300x200.png?text=Analysis+Preview", use_column_width=True)
        if st.button("Go to Real-Time Analysis"):
            st.session_state['page'] = 'realtime'
            st.rerun()
    
    with col3:
        st.markdown("<h3 style='color: grey;'>Vandalism Detection</h3>", unsafe_allow_html=True)
        st.image("https://via.placeholder.com/300x200.png?text=Detection+Preview", use_column_width=True)
        if st.button("Go to Detect Vandalism"):
            st.session_state['page'] = 'detect'
            st.rerun()

# Live Feed Page
def live():
    st.markdown("<h1 style='text-align: center; color: white;'>Live Security Feed</h1>", unsafe_allow_html=True)

    # Embed the Flask video feed in an iframe
    st.markdown("""
    <iframe src="http://localhost:5000/video_feed" width="700" height="500" frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

    # Additional controls can be added here
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("<h3 style='color: grey;'>Controls</h3>", unsafe_allow_html=True)
        if st.button("Start Stream"):
            st.write("Stream started")
        if st.button("Stop Stream"):
            st.write("Stream stopped")

frame_buffer = deque(maxlen=60)  # Buffer of 60 frames (~2 seconds at 30 FPS)

def video_capture(stframe):
    cap = cv2.VideoCapture(0)  # Change 0 to IP stream URL if needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec for saving
    out = None  # VideoWriter object for recording

    recording = False
    buffer_end = 0  # Counter for buffer time after detection ends

    while cap.isOpened() and st.session_state.streaming:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame)  # Store the frame in the buffer

        # Get prediction (int value)
        preds = process_video_frame(frame)

        # Map predicted label to class name
        label_map = {0: "Spitting", 1: "Graffiti", 2: "No Graffiti"}
        label = label_map[preds]

        # Add text to frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame in the Streamlit app
        stframe.image(frame, channels="RGB", use_column_width=True)

        # Start recording if graffiti or spit is detected
        if label in ["Spitting", "Graffiti"]:
            if not recording:
                # Start recording video and include buffered frames
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter(f'detection_{timestamp}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                
                for buffered_frame in frame_buffer:  # Write buffered frames to the file
                    out.write(cv2.cvtColor(buffered_frame, cv2.COLOR_RGB2BGR))
                
                recording = True
                buffer_end = 0

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write the current frame to the video file

        elif recording:
            # Keep recording for 2 seconds after detection ends
            if buffer_end < 60:  # 60 frames (~2 seconds)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write frame
                buffer_end += 1
            else:
                out.release()  # Stop recording
                recording = False

        time.sleep(0.03)

    if recording and out:
        out.release()  # Ensure video is saved when exiting

    cap.release()
# Real-Time Analysis Page
def realtime():
    st.markdown("<h1 style='text-align: center; color: white;'>Real-Time Security Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>This page contains real-time analytics and data visualization of security metrics.</p>", unsafe_allow_html=True)
    
    # Placeholder for real-time charts
    chart_data = [
        {"metric": "Motion Detected", "value": 15},
        {"metric": "Suspicious Activity", "value": 3},
        {"metric": "System Health", "value": 98},
    ]
    
    col1, col2, col3 = st.columns(3)
    for i, data in enumerate(chart_data):
        with [col1, col2, col3][i]:
            st.metric(label=data["metric"], value=data["value"])

# Detect Vandalism Page
def detect():
    st.markdown("<h1 style='text-align: center; color: white;'>Vandalism Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>This page contains the vandalism detection interface and results.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("https://via.placeholder.com/600x400.png?text=Vandalism+Detection+Feed", use_column_width=True)
    with col2:
        st.markdown("<h3 style='color: grey;'>Detection Log</h3>", unsafe_allow_html=True)
        st.text("10:15 AM - No vandalism detected")
        st.text("10:30 AM - Suspicious activity observed")
        st.text("10:45 AM - Area clear")

# Main app logic
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    
    if not st.session_state['logged_in']:
        if st.session_state['page'] == 'login':
            login()
        elif st.session_state['page'] == 'register':
            register()
    else:
        sidebar()
        if st.session_state['page'] == 'home':
            home()
        elif st.session_state['page'] == 'live':
            live()
        elif st.session_state['page'] == 'realtime':
            realtime()
        elif st.session_state['page'] == 'detect':
            detect()

if __name__ == "__main__":
    main()