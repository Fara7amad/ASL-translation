# ASL-translation
<h2> Introduction </h2>

This project aims to develop a machine learning model for translating real-time streaming American Sign Language (ASL) letters into corresponding English alphabets. This is an essential first step towards creating a comprehensive ASL translation tool that can assist in bridging the communication gap for individuals who rely on sign language. Our initial focus is on recognizing real-time hand signs representing the 26 letters of the English alphabet, achieving an accuracy rate of 70%.

![ASL_Alphabet](https://github.com/Fara7amad/ASL-translation/assets/106997246/ad4ec92e-41d0-476c-aa46-a1e1af67e2b6)


---
Table of Contents:

  1. Project Overview
  2. Model Training
  3. Limitations
  4. Results
  5. Conclusion
---

<h2>Project Overview</h2>

The project consists of several key components:

  1. Data collection and preprocessing
  2. Model architecture selection and training
  3. Evaluation and testing
  4. Application interface for real-time ASL letter detection
---
<h2>Model Training</h2>

The model was trained using a Convolutional Neural Network (CNN) architecture. Below is a brief overview of the training process:

  - Data Collection:
      - Collected images of individual letters from publicly available ASL datasets. [dataset link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
      - Preprocessed images to ensure uniformity (resizing, normalization).
      - Using MediaPipe for Hand Landmark Detection:
          MediaPipe is utilized for hand landmark detection, which is a crucial part of translating ASL letters. MediaPipe is a versatile framework that provides pre-trained models for various computer vision tasks, including hand tracking and landmark detection. Here’s an overview of how MediaPipe is employed in our project:
        
        1. MediaPipe Setup:
           - To get started, we initialize the MediaPipe Hands solution and the drawing utilities. MediaPipe Hands is configured to operate in static image mode, detect a maximum of one hand per image, and use a confidence threshold of 0.7 for hand detection.
            - Process Image Function:
              
                We created a function called process_image, which takes an image file path as input. This function performs the following steps:
              1. Reads the image using OpenCV.
              2. Converts the image from BGR to RGB format since MediaPipe expects RGB input.
              3. Processes the image using MediaPipe Hands to detect hand landmarks.
              4. If landmarks are detected, they are drawn on the original image, and their coordinates are collected in a list.

        3. Extract Landmarks Function:
              The extract_landmarks function processes an image to detect hand landmarks and returns a flattened array of these coordinates. If no landmarks are detected, it returns an array of zeros. This ensures a consistent output format for all images, with each image represented by a 63-element array corresponding to 21 landmarks with x, y, and z coordinates.
        4. Creating Dataset:
           
             To create our dataset, we developed a function that processes directories of images. Each subdirectory corresponds to a class of images representing different ASL letters. The function:
           1. Processes each image to extract hand landmarks using the extract_landmarks function.
           2. Labels each set of landmarks according to its class.
           3. aves the data to CSV files for later use in training and evaluation.

          This approach allows us to systematically collect and preprocess data, facilitating the training of our machine learning model for ASL letter recognition.

  - Model Architecture:
      - Utilized a CNN for its effectiveness in image recognition tasks.
      - Implemented layers: Convolutional layers, Pooling layers, Fully connected layers, and Dropout for regularization.

  - Training:
      - Split the dataset into training and validation sets.
      - Trained the model for 50 epochs with a batch size of 32.
      - Used cross-entropy loss and Adam optimizer.

  - Evaluation:
      - Evaluated model performance on a small test dataset.
---
<h2>Limitations</h2>

While the project demonstrates promising results, it is essential to acknowledge its limitations:

  - No Words Datasets:
        The current model only detects individual letters and cannot recognize words or sentences.

  - Movement Detection:
        Letters 'Z' and 'J' involve movement which our model currently does not detect accurately.

  - Small Test Dataset:
        The test dataset used for evaluation was small, which may affect the generalizability of the results.
---
<h2> Results </h2>

The model achieved an accuracy rate of 70%. This indicates a reasonable performance for initial attempts but also highlights the need for further improvements, especially in handling letters that involve movement and expanding the dataset for more robust training and evaluation.

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/4c4c6469-16fd-45ff-8053-4447ec5b0ee5)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/8363d827-6da7-4e30-a873-7dd1c0beeecb)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/b4ae5585-9852-4e3c-a5a0-3b481d311fea)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/8bee1cd3-6044-4889-baa9-ab0ff3e3caa8)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/d09aa3aa-6996-4f1a-a97c-ca9c8a300942)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/e331bcfe-fd05-446c-8de1-c9663f68f6bd)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/2285677d-a554-4db8-9422-18d372cdd997)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/5ceab8fb-136b-4264-8ad9-994c302cecc5)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/31a1e31d-3e3a-4cc4-b20a-6a60c319c6b5)

![image](https://github.com/Fara7amad/ASL-translation/assets/106997246/f9bf9af4-9436-4c62-85e4-5deca15bfbef)

---
<h2>Conclusion</h2>

This ASL translation project is a stepping stone towards more sophisticated ASL recognition systems. While the current model successfully detects real-time streaming hand signs for individual letters with a 70% accuracy, future work will focus on addressing the limitations such as detecting movements, increasing the dataset size, and expanding the model's capability to recognize words and sentences. With these enhancements, we aim to provide a more comprehensive tool to aid communication for the ASL community.
