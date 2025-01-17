
# HUMAN-POSE-ESTIMATION-USING-ML

### Overview

This project is developed as part of the **Microsoft & SAP AICTE Internship** focusing on **AI Technologies**. The internship is designed to enhance foundational AI skills and provide real-world project experience. The objective of this project is to implement human pose estimation using **OpenCV** and **Streamlit**, enabling users to detect and visualize human poses in images.

The system utilizes a pre-trained deep learning model to detect 19 key body parts in an image, such as the head, shoulders, elbows, knees, and more. These body parts are then connected to form a skeleton-like structure that visually represents human poses. The project leverages **OpenCV** for image processing and pose visualization, while **Streamlit** provides an easy-to-use web interface for users to interact with the system.

## Table of Contents

1. [Objective](#objective)
2. [Technologies Used](#technologies-used)
3. [How to Run](#how-to-run)
4. [Project Structure](#project-structure)
5. [Snapshots](#snapshots)
6. [GitHub Link for Code](#github-link-for-code)
7. [Licenses](#licenses)

## Objective

The primary objective of this project is to demonstrate human pose estimation by detecting 19 key body parts in an image and visualizing the poses. This can be used in applications such as fitness tracking, motion analysis, and gesture recognition.

Key functionalities of the project include:
- Uploading an image via a Streamlit-based interface.
- Detecting human poses using a pre-trained TensorFlow model.
- Visualizing the body parts and their connections in the form of a skeleton.
- Allowing users to adjust the detection threshold to fine-tune the pose detection.

The project helps in understanding the process of human pose estimation and provides an easy-to-implement framework for pose detection in various real-world applications.

## Technologies Used

- **OpenCV**: For image processing and drawing the detected poses.
- **Streamlit**: For creating a simple web interface where users can upload images and view pose estimates.
- **TensorFlow**: For using the pre-trained deep learning model to detect human poses.
- **NumPy**: For handling image data and matrix operations.
- **Python**: The primary programming language used for implementing the solution.

## How to Run

Follow the steps below to run the project on your local machine.

### 1. Clone the Repository

First, clone the repository from GitHub:

```bash
git clone https://github.com/NAIDU0019/HUMAN-POSE-ESTIMATION-USING-ML.git
cd HUMAN-POSE-ESTIMATION-USING-ML
```

### 2. Set Up a Virtual Environment

Create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Once the dependencies are installed, run the Streamlit apk:

```bash
streamlit run apk.py
```

This will open the Streamlit app in your browser.

### 5. Upload an Image

- After the app starts, you can upload an image by clicking the **"Upload an image"** button.
- You can adjust the detection threshold using the slider to fine-tune the detection accuracy.

The application will process the image, detect human poses, and display the result with visualized body parts connected by lines.

## Project Structure

The project contains the following files and directories:

```
HUMAN-POSE-ESTIMATION-USING-ML/
│
├── apk.py                # Streamlit app that runs the pose estimation logic
├── graph_opt.pb          # Pre-trained pose estimation model
├── requirements.txt      # List of required Python packages
├── images/               # Folder for storing snapshots and images
│   ├── original_image.jpg
│   └── pose_estimated.jpg
└── README.md             # This README file
```

- **app.py**: Contains the Streamlit code for uploading images, processing them, and displaying the results.
- **graph_opt.pb**: A TensorFlow pre-trained model file for human pose estimation.
- **requirements.txt**: Lists all the Python dependencies required for the project.
- **images/**: A folder where sample images and results can be stored.

## Snapshots

### Snapshot 1: Original Image
![image](https://github.com/user-attachments/assets/ea2c8eb9-93e6-4cb0-aac8-bcdaf5598d76)

This snapshot shows the image that is uploaded by the user. The pose estimation model will process this image to detect human poses.

### Snapshot 2: Pose Estimated Image
![image](https://github.com/user-attachments/assets/6a766ac5-1343-4f13-b2a0-da8f5c1dab4a)


This snapshot shows the result after pose estimation. The key body parts are detected and connected with lines to form a skeleton, representing the human pose in the image.

## GitHub Link for Code

You can access the full code for this project on GitHub:

[Pose Estimation Project Repository](https://github.com/NAIDU0019/HUMAN-POSE-ESTIMATION-USING-ML)

Feel free to fork the repository, clone it to your local machine, and contribute to the project!

## Licenses

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

