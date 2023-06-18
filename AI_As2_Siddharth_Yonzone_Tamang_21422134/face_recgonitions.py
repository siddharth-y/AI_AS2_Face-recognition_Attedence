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

            cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)  # Draw a rectangle around the face
            cv2.rectangle(output_image, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)  # Draw a filled rectangle for text background
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




# import cv2
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime, date
# import pyttsx3 as textSpeech
# import tkinter as tk
# from PIL import Image, ImageTk

# engine = textSpeech.init()

# # Initialize Tkinter window
# window = tk.Tk()
# window.configure(bg="white")

# # Create a label to display the webcam feed
# webcam_frame = tk.Label(window, bg="white")
# webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# # Create labels for student information and attendance status
# std_info_lable = tk.Label(window, text="Student Information", bg="white")
# atted_label = tk.Label(window, text="Attendance Status", bg="white")

# # Create text boxes to display student information and attendance status
# std_info_txt = tk.Text(window, height=10, width=30)
# atted_txt = tk.Text(window, height=10, width=30)

# # Grid layout for labels and text boxes
# std_info_lable.grid(row=0, column=1, padx=10, pady=10)
# atted_label.grid(row=0, column=2, padx=10, pady=10)
# std_info_txt.grid(row=1, column=1, padx=10, pady=10)
# atted_txt.grid(row=1, column=2, padx=10, pady=10)

# # Load student image dataset and face encodings
# path = 'image'
# std_img = []
# std_name = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     std_img.append(curImg)
#     std_name.append(os.path.splitext(cl)[0])

# def findEncoding(image):
#     encoding_img = []
#     for img in image:
#         if img is None:
#             continue
#         # Resize the image for better face recognition accuracy
#         img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoding = face_recognition.face_encodings(img)[0]
#         encoding_img.append(encoding)
#     return encoding_img

# encode_list = findEncoding(std_img)

# vid = cv2.VideoCapture(0)

# def show_frame():
#     _, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize the frame

#     faces_in_frame = face_recognition.face_locations(frame)
#     encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)

#     output_image = frame.copy()

#     recognized_names = []  # List to store recognized names

#     for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
#         matches = face_recognition.compare_faces(encode_list, encodeFace)
#         face_distances = face_recognition.face_distance(encode_list, encodeFace)

#         if len(face_distances) > 0:
#             match_indices = np.where(matches)[0]
#             recognized_names.extend([std_name[i].upper() for i in match_indices])

#             top, right, bottom, left = faceLoc
#             top *= 1
#             right *= 1
#             bottom *= 1
#             left *= 1

#             cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.rectangle(output_image, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(output_image, ", ".join(recognized_names), (left + 4, bottom - 4),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     # Update attendance status for recognized names
#     for name in recognized_names:
#         mark_attendance(name)

#     img = Image.fromarray(output_image)
#     img = ImageTk.PhotoImage(image=img)
#     webcam_frame.img = img
#     webcam_frame.configure(image=img)
#     webcam_frame.after(10, show_frame)

# def mark_attendance(name):
#     # Generate the current date
#     now = datetime.now()
#     datestr = now.strftime('%Y-%m-%d')

#     # Create a new CSV file for the current date if it doesn't exist
#     filename = f'attendance_{datestr}.csv'
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Name, Date, Time\n")

#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         name_list = [entry.split(',')[0].strip() for entry in data_list]

#     timestr = now.strftime('%H:%M')

#     if name not in name_list:
#         with open(filename, 'a') as f:
#             f.write(f'{name}, {datestr}, {timestr}\n')
#         statement = f'Welcome to class, {name}'
#         engine.say(statement)
#         engine.runAndWait()
#         status = f"{name}: Present"
#     else:
#         status = f"{name}: Already Marked"

#     # Get all attendance names for display
#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         names = [entry.split(',')[0].strip() for entry in data_list]

#     update_student_info(names, status)

# # Function to update student information and attendance status
# def update_student_info(names, status):
#     # Update student information text box
#     std_info_txt.delete("1.0", tk.END)
#     std_info_txt.insert(tk.END, "\n".join(names))

#     # Update attendance status text box
#     atted_txt.delete("1.0", tk.END)
#     atted_txt.insert(tk.END, status)

# # Run the webcam feed and attendance update
# show_frame()

# # Run GUI main loop
# window.mainloop()

# cv2.destroyAllWindows()



# import cv2
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime, date
# import pyttsx3 as textSpeech
# import tkinter as tk
# import tensorflow as tf
# from PIL import Image, ImageTk
# from tensorflow.keras.preprocessing.image import img_to_array

# engine = textSpeech.init()

# # Load the trained model
# model = tf.keras.models.load_model('Script/trained_model.h5')

# # Initialize Tkinter window
# window = tk.Tk()
# window.configure(bg="white")

# # Create a label to display the webcam feed
# webcam_frame = tk.Label(window, bg="white")
# webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# # Create labels for student information and attendance status
# std_info_label = tk.Label(window, text="Student Information", bg="white")
# atted_label = tk.Label(window, text="Attendance Status", bg="white")

# # Create text boxes to display student information and attendance status
# std_info_txt = tk.Text(window, height=10, width=30)
# atted_txt = tk.Text(window, height=10, width=30)

# # Grid layout for labels and text boxes
# std_info_label.grid(row=0, column=1, padx=10, pady=10)
# atted_label.grid(row=0, column=2, padx=10, pady=10)
# std_info_txt.grid(row=1, column=1, padx=10, pady=10)
# atted_txt.grid(row=1, column=2, padx=10, pady=10)

# # Load student image dataset and face encodings
# path = 'Script/Dataset'
# std_img = []
# std_name = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     std_img.append(curImg)
#     std_name.append(os.path.splitext(cl)[0])

# encode_list = []
# for img in std_img:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     face_encodings = face_recognition.face_encodings(img)[0]
#     encode_list.append(face_encodings)

# vid = cv2.VideoCapture(0)

# def show_frame():
#     _, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces_in_frame = face_recognition.face_locations(frame)
#     encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)

#     output_image = frame.copy()

#     recognized_names = []  # List to store recognized names

#     for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
#         matches = face_recognition.compare_faces(encode_list, encodeFace)
#         face_distances = face_recognition.face_distance(encode_list, encodeFace)

#         if len(face_distances) > 0:
#             match_indices = np.where(matches)[0]
#             recognized_names.extend([std_name[i].upper() for i in match_indices])

#             top, right, bottom, left = faceLoc
#             top *= 2
#             right *= 2
#             bottom *= 2
#             left *= 2

#             cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 3)
#             cv2.rectangle(output_image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(output_image, ", ".join(recognized_names), (left + 6, bottom - 6),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     # Update attendance status for recognized names
#     for name in recognized_names:
#         mark_attendance(name)

#     img = Image.fromarray(output_image)
#     img = ImageTk.PhotoImage(image=img)
#     webcam_frame.img = img
#     webcam_frame.configure(image=img)
#     webcam_frame.after(10, show_frame)

# def mark_attendance(name):
#     # Generate the current date
#     now = datetime.now()
#     datestr = now.strftime('%Y-%m-%d')

#     # Create a new CSV file for the current date if it doesn't exist
#     filename = f'attendance_{datestr}.csv'
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Name, Date, Time\n")

#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         name_list = [entry.split(',')[0].strip() for entry in data_list]

#     timestr = now.strftime('%H:%M')

#     if name not in name_list:
#         with open(filename, 'a') as f:
#             f.write(f'{name}, {datestr}, {timestr}\n')
#         statement = f'Welcome to class, {name}'
#         engine.say(statement)
#         engine.runAndWait()
#         status = f"{name}: Present"
#     else:
#         status = f"{name}: Already Marked"

#     # Get all attendance names for display
#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         names = [entry.split(',')[0].strip() for entry in data_list]

#     update_student_info(names, status)

# # Function to update student information and attendance status
# def update_student_info(names, status):
#     # Update student information text box
#     std_info_txt.delete("1.0", tk.END)
#     std_info_txt.insert(tk.END, "\n".join(names))

#     # Update attendance status text box
#     atted_txt.delete("1.0", tk.END)
#     atted_txt.insert(tk.END, status)

# # Run the webcam feed and attendance update
# show_frame()

# # Run GUI main loop
# window.mainloop()

# cv2.destroyAllWindows()



# import cv2
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime, date
# import pyttsx3 as textSpeech
# import tkinter as tk
# from PIL import Image, ImageTk
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# engine = textSpeech.init()

# # Initialize Tkinter window
# window = tk.Tk()
# window.configure(bg="white")

# # Create a label to display the webcam feed
# webcam_frame = tk.Label(window, bg="white")
# webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# # Create labels for student information and attendance status
# std_info_lable = tk.Label(window, text="Student Information", bg="white")
# atted_label = tk.Label(window, text="Attendance Status", bg="white")

# # Create text boxes to display student information and attendance status
# std_info_txt = tk.Text(window, height=10, width=30)
# atted_txt = tk.Text(window, height=10, width=30)

# # Grid layout for labels and text boxes
# std_info_lable.grid(row=0, column=1, padx=10, pady=10)
# atted_label.grid(row=0, column=2, padx=10, pady=10)
# std_info_txt.grid(row=1, column=1, padx=10, pady=10)
# atted_txt.grid(row=1, column=2, padx=10, pady=10)

# # Load student image dataset and face encodings
# path = 'image'
# std_img = []
# std_name = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     std_img.append(curImg)
#     std_name.append(os.path.splitext(cl)[0])

# # Augmentation parameters
# num_augmented_images = 10  # Number of augmented images to generate for each original image

# # Create the augmentation generator
# augmentation = ImageDataGenerator(
#     rotation_range=45,
#     shear_range=16,
#     horizontal_flip=True,
#     vertical_flip=True,
#     # Add more augmentation options as needed
# )

# # Generate augmented images and add them to the student image dataset
# augmented_std_img = []
# augmented_std_name = []
# for i, image in enumerate(std_img):
#     image = cv2.resize(image, (150, 150))  # Resize the image to a consistent shape
#     image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size
#     j = 0
#     for augmented_image in augmentation.flow(image, batch_size=1):
#         augmented_image = augmented_image[0]  # Retrieve the augmented image
#         augmented_image = cv2.resize(augmented_image, (150, 150))  # Resize the augmented image
#         augmented_std_img.append(augmented_image)
#         augmented_std_name.append(std_name[i] + f"_augmented_{j}")
#         j += 1
#         if j >= num_augmented_images:
#             break

# # Convert augmented_std_img to a numpy array
# augmented_std_img = np.array(augmented_std_img)

# # Check the size of std_img
# if len(std_img) != len(std_name):
#     print("Error: std_img and std_name arrays should have the same size")
#     exit()

# # Reshape std_img to match the shape of augmented_std_img
# std_img = np.reshape(std_img, (len(std_img), 150, 150, 3))

# # Combine original and augmented images and their respective names
# std_img = np.concatenate((std_img, augmented_std_img), axis=0)
# std_name = std_name + augmented_std_name



# def findEncoding(image):
#     encoding_img = []
#     for img in image:
#         if img is None:
#             continue
#         img = resize(img, 0.50)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoding = face_recognition.face_encodings(img)[0]
#         encoding_img.append(encoding)
#     return encoding_img


# def resize(img, size):
#     width = int(img.shape[1] * size)
#     height = int(img.shape[0] * size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


# encode_list = findEncoding(std_img)

# vid = cv2.VideoCapture(0)


# def show_frame():
#     _, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = resize(frame, 0.5)  # Resize the frame

#     faces_in_frame = face_recognition.face_locations(frame)
#     encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)

#     output_image = frame.copy()

#     recognized_names = []  # List to store recognized names

#     for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
#         matches = face_recognition.compare_faces(encode_list, encodeFace)
#         face_distances = face_recognition.face_distance(encode_list, encodeFace)

#         if len(face_distances) > 0:
#             match_indices = np.where(matches)[0]
#             recognized_names.extend([std_name[i].upper() for i in match_indices])

#             top, right, bottom, left = faceLoc
#             top *= 2
#             right *= 2
#             bottom *= 2
#             left *= 2

#             cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 3)
#             cv2.rectangle(output_image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(output_image, ", ".join(recognized_names), (left + 6, bottom - 6),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     # Update attendance status for recognized names
#     for name in recognized_names:
#         mark_attendance(name)

#     img = Image.fromarray(output_image)
#     img = ImageTk.PhotoImage(image=img)
#     webcam_frame.img = img
#     webcam_frame.configure(image=img)
#     webcam_frame.after(10, show_frame)


# def mark_attendance(name):
#     # Generate the current date
#     now = datetime.now()
#     datestr = now.strftime('%Y-%m-%d')

#     # Create a new CSV file for the current date if it doesn't exist
#     filename = f'attendance_{datestr}.csv'
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Name, Date, Time\n")

#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         name_list = [entry.split(',')[0].strip() for entry in data_list]

#     timestr = now.strftime('%H:%M')

#     if name not in name_list:
#         with open(filename, 'a') as f:
#             f.write(f'{name}, {datestr}, {timestr}\n')
#         statement = f'Welcome to class, {name}'
#         engine.say(statement)
#         engine.runAndWait()
#         status = f"{name}: Present"
#     else:
#         status = f"{name}: Already Marked"

#     # Get all attendance names for display
#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         names = [entry.split(',')[0].strip() for entry in data_list]

#     update_student_info(names, status)


# # Function to update student information and attendance status
# def update_student_info(names, status):
#     # Update student information text box
#     std_info_txt.delete("1.0", tk.END)
#     std_info_txt.insert(tk.END, "\n".join(names))

#     # Update attendance status text box
#     atted_txt.delete("1.0", tk.END)
#     atted_txt.insert(tk.END, status)


# # Run the webcam feed and attendance update
# show_frame()

# # Run GUI main loop
# window.mainloop()

# cv2.destroyAllWindows()



# import cv2
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime, date
# import pyttsx3 as textSpeech
# import tkinter as tk
# from PIL import Image, ImageTk
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# engine = textSpeech.init()

# # Initialize Tkinter window
# window = tk.Tk()
# window.configure(bg="white")

# # Create a label to display the webcam feed
# webcam_frame = tk.Label(window, bg="white")
# webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# # Create labels for student information and attendance status
# std_info_lable = tk.Label(window, text="Student Information", bg="white")
# atted_label = tk.Label(window, text="Attendance Status", bg="white")

# # Create text boxes to display student information and attendance status
# std_info_txt = tk.Text(window, height=10, width=30)
# atted_txt = tk.Text(window, height=10, width=30)

# # Grid layout for labels and text boxes
# std_info_lable.grid(row=0, column=1, padx=10, pady=10)
# atted_label.grid(row=0, column=2, padx=10, pady=10)
# std_info_txt.grid(row=1, column=1, padx=10, pady=10)
# atted_txt.grid(row=1, column=2, padx=10, pady=10)

# # Load student image dataset and face encodings
# path = 'image'
# std_img = []
# std_name = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     std_img.append(curImg)
#     std_name.append(os.path.splitext(cl)[0])

# # Augmentation parameters
# num_augmented_images = 10  # Number of augmented images to generate for each original image

# # Create the augmentation generator
# augmentation = ImageDataGenerator(
#     rotation_range=45,
#     shear_range=16,
#     horizontal_flip=True,
#     vertical_flip=True,
#     # Add more augmentation options as needed
# )

# # Generate augmented images and add them to the student image dataset
# augmented_std_img = []
# augmented_std_name = []
# for i, image in enumerate(std_img):
#     image = cv2.resize(image, (150, 150))  # Resize the image to a consistent shape
#     image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size
#     j = 0
#     for augmented_image in augmentation.flow(image, batch_size=1):
#         augmented_image = augmented_image[0]  # Retrieve the augmented image
#         augmented_image = cv2.resize(augmented_image, (150, 150))  # Resize the augmented image
#         augmented_std_img.append(augmented_image)
#         augmented_std_name.append(std_name[i] + f"_augmented_{j}")
#         j += 1
#         if j >= num_augmented_images:
#             break

# # Combine original and augmented images and their respective names
# std_img = np.concatenate((std_img, augmented_std_img), axis=0)
# std_name = std_name + augmented_std_name


# def findEncoding(image):
#     encoding_img = []
#     for img in image:
#         if img is None:
#             continue
#         img = resize(img, 0.50)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoding = face_recognition.face_encodings(img)[0]
#         encoding_img.append(encoding)
#     return encoding_img

# def resize(img, size):
#     width = int(img.shape[1] * size)
#     height = int(img.shape[0] * size)
#     dimention = (width, height)
#     return cv2.resize(img, dimention, interpolation=cv2.INTER_AREA)

# encode_list = findEncoding(std_img)

# vid = cv2.VideoCapture(0)

# def show_frame():
#     _, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = resize(frame, 0.5)  # Resize the frame

#     faces_in_frame = face_recognition.face_locations(frame)
#     encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)

#     output_image = frame.copy()

#     recognized_names = []  # List to store recognized names

#     for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
#         matches = face_recognition.compare_faces(encode_list, encodeFace)
#         face_distances = face_recognition.face_distance(encode_list, encodeFace)

#         if len(face_distances) > 0:
#             match_indices = np.where(matches)[0]
#             recognized_names.extend([std_name[i].upper() for i in match_indices])

#             top, right, bottom, left = faceLoc
#             top *= 2
#             right *= 2
#             bottom *= 2
#             left *= 2

#             cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 3)
#             cv2.rectangle(output_image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(output_image, ", ".join(recognized_names), (left + 6, bottom - 6),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     # Update attendance status for recognized names
#     for name in recognized_names:
#         mark_attendance(name)

#     img = Image.fromarray(output_image)
#     img = ImageTk.PhotoImage(image=img)
#     webcam_frame.img = img
#     webcam_frame.configure(image=img)
#     webcam_frame.after(10, show_frame)

# def mark_attendance(name):
#     # Generate the current date
#     now = datetime.now()
#     datestr = now.strftime('%Y-%m-%d')

#     # Create a new CSV file for the current date if it doesn't exist
#     filename = f'attendance_{datestr}.csv'
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Name, Date, Time\n")

#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         name_list = [entry.split(',')[0].strip() for entry in data_list]

#     timestr = now.strftime('%H:%M')

#     if name not in name_list:
#         with open(filename, 'a') as f:
#             f.write(f'{name}, {datestr}, {timestr}\n')
#         statement = f'Welcome to class, {name}'
#         engine.say(statement)
#         engine.runAndWait()
#         status = f"{name}: Present"
#     else:
#         status = f"{name}: Already Marked"

#     # Get all attendance names for display
#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         names = [entry.split(',')[0].strip() for entry in data_list]

#     update_student_info(names, status)

# # Function to update student information and attendance status
# def update_student_info(names, status):
#     # Update student information text box
#     std_info_txt.delete("1.0", tk.END)
#     std_info_txt.insert(tk.END, "\n".join(names))

#     # Update attendance status text box
#     atted_txt.delete("1.0", tk.END)
#     atted_txt.insert(tk.END, status)

# # Run the webcam feed and attendance update
# show_frame()

# # Run GUI main loop
# window.mainloop()

# cv2.destroyAllWindows()



# import cv2
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime, date
# import pyttsx3 as textSpeech
# import tkinter as tk
# from PIL import Image, ImageTk

# engine = textSpeech.init()

# # Initialize Tkinter window
# window = tk.Tk()
# window.configure(bg="white")

# # Create a label to display the webcam feed
# webcam_frame = tk.Label(window, bg="white")
# webcam_frame.grid(row=0, column=0, padx=10, pady=10)

# # Create labels for student information and attendance status
# std_info_lable = tk.Label(window, text="Student Information", bg="white")
# atted_label = tk.Label(window, text="Attendance Status", bg="white")

# # Create text boxes to display student information and attendance status
# std_info_txt = tk.Text(window, height=10, width=30)
# atted_txt = tk.Text(window, height=10, width=30)

# # Grid layout for labels and text boxes
# std_info_lable.grid(row=0, column=1, padx=10, pady=10)
# atted_label.grid(row=0, column=2, padx=10, pady=10)
# std_info_txt.grid(row=1, column=1, padx=10, pady=10)
# atted_txt.grid(row=1, column=2, padx=10, pady=10)

# # Load student image dataset and face encodings
# path = 'image'
# std_img = []
# std_name = []
# myList = os.listdir(path)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     std_img.append(curImg)
#     std_name.append(os.path.splitext(cl)[0])

# def findEncoding(image):
#     encoding_img = []
#     for img in image:
#         if img is None:
#             continue
#         img = resize(img, 0.50)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoding = face_recognition.face_encodings(img)[0]
#         encoding_img.append(encoding)
#     return encoding_img

# def resize(img, size):
#     width = int(img.shape[1] * size)
#     height = int(img.shape[0] * size)
#     dimention = (width, height)
#     return cv2.resize(img, dimention, interpolation=cv2.INTER_AREA)

# encode_list = findEncoding(std_img)

# vid = cv2.VideoCapture(0)

# def show_frame():
#     _, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = resize(frame, 0.5)  # Resize the frame

#     faces_in_frame = face_recognition.face_locations(frame)
#     encode_in_frame = face_recognition.face_encodings(frame, faces_in_frame)

#     output_image = frame.copy()

#     recognized_names = []  # List to store recognized names

#     for encodeFace, faceLoc in zip(encode_in_frame, faces_in_frame):
#         matches = face_recognition.compare_faces(encode_list, encodeFace)
#         face_distances = face_recognition.face_distance(encode_list, encodeFace)

#         if len(face_distances) > 0:
#             match_indices = np.where(matches)[0]
#             recognized_names.extend([std_name[i].upper() for i in match_indices])

#             top, right, bottom, left = faceLoc
#             top *= 2
#             right *= 2
#             bottom *= 2
#             left *= 2

#             cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.rectangle(output_image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(output_image, ", ".join(recognized_names), (left + 6, bottom - 6),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#     # Update attendance status for recognized names
#     for name in recognized_names:
#         mark_attendance(name)

#     img = Image.fromarray(output_image)
#     img = ImageTk.PhotoImage(image=img)
#     webcam_frame.img = img
#     webcam_frame.configure(image=img)
#     webcam_frame.after(10, show_frame)

# def mark_attendance(name):
#     # Generate the current date
#     now = datetime.now()
#     datestr = now.strftime('%Y-%m-%d')

#     # Create a new CSV file for the current date if it doesn't exist
#     filename = f'attendance_{datestr}.csv'
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Name, Date, Time\n")

#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         name_list = [entry.split(',')[0].strip() for entry in data_list]

#     timestr = now.strftime('%H:%M')

#     if name not in name_list:
#         with open(filename, 'a') as f:
#             f.write(f'{name}, {datestr}, {timestr}\n')
#         statement = f'Welcome to class, {name}'
#         engine.say(statement)
#         engine.runAndWait()
#         status = f"{name}: Present"
#     else:
#         status = f"{name}: Already Marked"

#     # Get all attendance names for display
#     with open(filename, 'r') as f:
#         data_list = f.readlines()
#         names = [entry.split(',')[0].strip() for entry in data_list]

#     update_student_info(names, status)

# # Function to update student information and attendance status
# def update_student_info(names, status):
#     # Update student information text box
#     std_info_txt.delete("1.0", tk.END)
#     std_info_txt.insert(tk.END, "\n".join(names))

#     # Update attendance status text box
#     atted_txt.delete("1.0", tk.END)
#     atted_txt.insert(tk.END, status)

# # Run the webcam feed and attendance update
# show_frame()

# # Run GUI main loop
# window.mainloop()

# cv2.destroyAllWindows()



