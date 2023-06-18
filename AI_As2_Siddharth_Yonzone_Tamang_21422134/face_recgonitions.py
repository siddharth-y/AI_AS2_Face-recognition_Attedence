import cv2  # Import OpenCV library for image and video processing
import face_recognition  # Import face recognition library
import numpy as np  # Import NumPy library for array operations
import csv  # Import CSV library for handling CSV files
import os  # Import os library for file and directory operations
from datetime import datetime, date  # Import datetime library for working with dates and times
import pyttsx3 as textSpeech  # Import pyttsx3 library for text-to-speech synthesis
import tkinter as tk  # Import Tkinter library for GUI
from PIL import Image, ImageTk  # Import PIL library for working with images

engine = textSpeech.init()  # Initialize text-to-speech engine

# Initialize Tkinter window
window = tk.Tk()
window.configure(bg="white")

# Create a label to display the webcam feed
webcam_frame = tk.Label(window, bg="white")
webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# Create labels for student information and attendance status
std_info_lable = tk.Label(window, text="Student Information", bg="white")
atted_label = tk.Label(window, text="Attendance Status", bg="white")

# Create text boxes to display student information and attendance status
std_info_txt = tk.Text(window, height=10, width=30)
atted_txt = tk.Text(window, height=10, width=30)

# Grid layout for labels and text boxes
std_info_lable.grid(row=0, column=1, padx=10, pady=10)
atted_label.grid(row=0, column=2, padx=10, pady=10)
std_info_txt.grid(row=1, column=1, padx=10, pady=10)
atted_txt.grid(row=1, column=2, padx=10, pady=10)

# Load student image dataset and face encodings
path = 'image'
std_img = []
std_name = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # Read image from file
    std_img.append(curImg)  # Append image to list
    std_name.append(os.path.splitext(cl)[0])  # Append name to list by removing file extension

def findEncoding(image):
    encoding_img = []
    for img in image:
        if img is None:
            continue
        # Resize the image for better face recognition accuracy
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
        encoding = face_recognition.face_encodings(img)[0]  # Get face encodings from the image
        encoding_img.append(encoding)  # Append face encodings to the list
    return encoding_img

encode_list = findEncoding(std_img)  # Get face encodings for the loaded student images

vid = cv2.VideoCapture(0)  # Initialize video capture from default webcam

def show_frame():
    _, frame = vid.read()  # Read a frame from the video capture
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR frame to RGB
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the frame

    faces_in_frame = face_recognition.face_locations(frame)  # Locate faces in the frame
    encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)  # Get face encodings from the frame

    output_image = frame.copy()  # Create a copy of the frame for output

    recognized_names = []  # List to store recognized names

    for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
        matches = face_recognition.compare_faces(encode_list, encodeFace)  # Compare face encodings with loaded encodings
        face_distances = face_recognition.face_distance(encode_list, encodeFace)  # Calculate face distances for recognition

        if len(face_distances) > 0:
            match_indices = np.where(matches)[0]  # Get the indices of recognized faces
            recognized_names.extend([std_name[i].upper() for i in match_indices])  # Append recognized names to the list

            top, right, bottom, left = faceLoc  # Get the face location coordinates
            top *= 1  # Scale the coordinates (optional)
            right *= 1  # Scale the coordinates (optional)
            bottom *= 1  # Scale the coordinates (optional)
            left *= 1  # Scale the coordinates (optional)

            cv2.rectangle(output_image, (left, top), (right, bottom), (0, 0, 255), 2)  # Draw a rectangle around the face
            cv2.rectangle(output_image, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)  # Draw a filled rectangle for text background
            cv2.putText(output_image, ", ".join(recognized_names), (left + 4, bottom - 4),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Display recognized names on the image

    # Update attendance status for recognized names
    for name in recognized_names:
        mark_attendance(name)

    img = Image.fromarray(output_image)  # Convert the output image array to PIL Image
    img = ImageTk.PhotoImage(image=img)  # Convert PIL Image to Tkinter Image
    webcam_frame.img = img  # Store the image reference to prevent garbage collection
    webcam_frame.configure(image=img)  # Configure the label to display the image
    webcam_frame.after(10, show_frame)  # Schedule the next frame update

def mark_attendance(name):
    # Generate the current date
    now = datetime.now()
    datestr = now.strftime('%Y-%m-%d')

    # Create a new CSV file for the current date if it doesn't exist
    filename = f'attendance_{datestr}.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name, Date, Time\n")

    with open(filename, 'r') as f:
        data_list = f.readlines()
        name_list = [entry.split(',')[0].strip() for entry in data_list]  # Get the list of names already marked

    timestr = now.strftime('%H:%M')  # Get the current time

    if name not in name_list:
        with open(filename, 'a') as f:
            f.write(f'{name}, {datestr}, {timestr}\n')  # Write the attendance record to the CSV file
        statement = f'Welcome to class, {name}'
        engine.say(statement)  # Speak the welcome message
        engine.runAndWait()
        status = f"{name}: Present"
    else:
        status = f"{name}: Already Marked"

    # Get all attendance names for display
    with open(filename, 'r') as f:
        data_list = f.readlines()
        names = [entry.split(',')[0].strip() for entry in data_list]  # Get all names in the attendance record

    update_student_info(names, status)  # Update student information and attendance status

# Function to update student information and attendance status
def update_student_info(names, status):
    # Update student information text box
    std_info_txt.delete("1.0", tk.END)  # Clear the text box
    std_info_txt.insert(tk.END, "\n".join(names))  # Insert names into the text box

    # Update attendance status text box
    atted_txt.delete("1.0", tk.END)  # Clear the text box
    atted_txt.insert(tk.END, status)  # Insert attendance status into the text box

# Run the webcam feed and attendance update
show_frame()

# Run GUI main loop
window.mainloop()

cv2.destroyAllWindows()  # Close all OpenCV windows


