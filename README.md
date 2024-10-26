# Face Mask Detection

**A Django-powered image processing application to detect whether a person is wearing a mask.**

## Live Link : [click here]{https://face-mask-detection-12.onrender.com/predict/}

## Overview

This project leverages the power of Django and computer vision techniques to accurately identify individuals wearing face masks in images. It's a practical solution for various applications, such as security systems, public health monitoring, and more.

## Features

  * **Real-time Image Processing:** Quickly analyze images and return accurate results.
  * **Robust Model:** A well-trained model capable of handling diverse face orientations and lighting conditions.
  * **User-Friendly Interface:** A simple and intuitive interface for easy interaction.
  * **Scalability:** Designed to handle large-scale image processing tasks.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-Myash21/face-mask-detection.git](https://github.com/your-Myash21/face-mask-detection.git)
    ```
2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Django Server:**
    ```bash
    python manage.py runserver
    ```

## How it Works

1.  **Image Upload:** Users can upload an image to the web application.
2.  **Image Preprocessing:** The uploaded image is preprocessed to enhance features and normalize the input.
3.  **Model Inference:** The preprocessed image is fed into a trained deep learning model, which predicts whether a face mask is present.
4.  **Result Display:** The application displays the prediction result, along with the processed image and confidence score.


## Future Improvements

  * **Real-time Video Analysis:** Extend the application to process video streams in real-time.
  * **Mobile Application:** Develop a mobile app for on-the-go mask detection.
  * **Edge Deployment:** Optimize the model for deployment on edge devices.

## Contributing

We welcome contributions to improve this project. Feel free to fork the repository, make changes, and submit a pull request.
