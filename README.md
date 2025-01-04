# 🌱 Plant Disease Detection Using AI

# 1. Project Overview
   
This project implements a Plant Disease Detection System using a trained Convolutional Neural Network (CNN) model to predict the disease present in a plant leaf based on an uploaded image. The web application built using Streamlit allows users to upload an image of a plant leaf, and the system predicts both the crop type and the disease affecting the plant.

---
# 2. Technologies Used

Streamlit: A Python library to build interactive web applications for data science and machine learning projects.
Keras: High-level neural networks API used to build and train deep learning models.
TensorFlow: Open-source machine learning library used for training and deploying the model.
Pickle: Python library to serialize the machine learning model into a file for easy loading and inference.
Pillow (PIL): Python Imaging Library used for image processing tasks.
NumPy: Library for numerical operations, particularly for image data manipulation.

---
# 3. Features
Image Upload: Users can upload plant leaf images in JPG, PNG, or JPEG formats.
AI-Based Prediction: The model predicts the disease and the crop based on the uploaded leaf image.
Instant Feedback: The app provides instant feedback with a progress bar while classifying the uploaded image.
User-Friendly Interface: Built with Streamlit, providing an easy-to-use and interactive interface for plant disease detection.
Disease Information: Displays detailed information about the predicted disease and the affected crop.

---
# 4. Getting Started
# 4.1 Clone the Repository
Start by cloning the project repository to your local machine:


Copy code
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
 # 4.2 Install Dependencies
Ensure you have Python 3.6 or higher installed. Install the required dependencies using the following command:

Copy code
pip install -r requirements.txt

 # 4.3 Prepare the Model
The application requires a pre-trained model to predict plant diseases. Ensure that the trained model file (plant_disease_model.pkl) is placed in the following directory:


Copy code
Model/plant_disease_model.pkl
If you do not have a trained model, you can either train one using relevant datasets or use an existing model. You can find various publicly available plant disease datasets to train the model.

# 4.4 Running the Application
To run the application, execute the following command in the project directory:

Copy code
streamlit run app.py
This will launch the application in your default web browser. You will be able to upload plant leaf images and receive predictions about the crop and disease.
---
# 5. Application Workflow
# 5.1 Uploading an Image
The user is prompted to upload a plant leaf image in JPG, PNG, or JPEG format.
Once the image is uploaded, it is displayed on the page, and the AI model starts the classification process.
# 5.2 Model Inference
The uploaded image is processed and passed through a pre-trained Convolutional Neural Network (CNN).
The model predicts the crop type and the disease affecting the plant leaf.
The result is displayed on the interface, including the crop name and disease name.
# 5.3 Progress Bar and Result Display
While the image is being processed, a progress bar is shown to the user to indicate the classification process.
After classification, the predicted crop and disease names are displayed, along with detailed information.
---
# 6. Project Structure
The project directory is structured as follows:

Copy code
plant-disease-detection/
│
├── app.py                       # Main Streamlit application file.
├── requirements.txt             # List of Python dependencies required for the project.
├── Model/                       # Directory containing the pre-trained model.
│   └── plant_disease_model.pkl  # Pre-trained model file.
├── temp/                        # Temporary folder to store uploaded images.
├── README.md                    # Project documentation (this file).
└── LICENSE                      # License information (MIT or other).
---
# 7. Model Details
The plant_disease_model.pkl is a Convolutional Neural Network (CNN) that has been trained to detect diseases in various plant crops based on leaf images. The model was trained on a publicly available plant disease dataset and can recognize diseases in a variety of crops such as tomatoes, potatoes, apples, and others.

Key model details:

Input Size: 224x224 pixels (resized from the original image).
Model Architecture: Convolutional Neural Network (CNN).
Framework Used: Keras with TensorFlow backend.
# 7.1 Training the Model
To train the model, you can use a plant disease dataset such as the PlantVillage Dataset. The model can be trained using the following steps:

Preprocess the images (resize, normalize, etc.).
Use a CNN architecture to extract features.
Train the model using labeled data (disease labels).
Save the trained model using pickle for later use.
---
# 8. Contributing
Contributions to improve the model, UI, or codebase are welcome! Here’s how you can contribute:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Make changes and commit them (git commit -am 'Add new feature').
Push your changes to the repository (git push origin feature-name).
Open a pull request to submit your changes.
Areas for Improvement
Model Accuracy: Training on additional datasets to improve prediction accuracy.
Support for more diseases: Adding more diseases to the model's capabilities.
User Interface: Improving the design of the application
