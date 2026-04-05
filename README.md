# Pneumonia-Detection-from-Chest-X-ray-Images

An AI-powered system that detects **Pneumonia from Chest X-Ray images** using **Deep Learning (CNN) and Numerical Optimization techniques**.
The system trains a neural network to classify chest X-ray images into **Normal** or **Pneumonia** and provides predictions through a web interface.

---

## Project Overview

Pneumonia is a lung infection that can be detected using chest X-ray images.
This project applies **Artificial Intelligence and Deep Learning** to automatically analyze X-ray images and assist in early diagnosis.

The system performs the following tasks:

* Preprocess chest X-ray images
* Train a Convolutional Neural Network (CNN)
* Apply optimization techniques during training
* Evaluate model performance
* Generate training graphs
* Provide predictions through a web interface

---

## Project Structure

```
pneumonia-ai-optimization/

├── README.md
├── requirements.txt
├── main.py
├── api.py

├── config/
│   └── config.py

├── data/
│   ├── dataset_loader.py
│   └── preprocessing.py

├── models/
│   └── cnn_model.py

├── optimizers/
│   └── lipschitz_momentum.py

├── training/
│   ├── train.py
│   └── evaluate.py

├── utils/
│   ├── metrics.py
│   └── plot_results.py

├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js

├── results/
│   ├── accuracy_plot.png
│   └── loss_plot.png

├── checkpoints/
│   └── model_weights.pth

└── notebooks/
    └── experiment_analysis.ipynb
```

---

## Dataset

Dataset used: **Chest X-Ray Pneumonia Dataset**

Dataset structure:

```
Dataset/
   chest_xray/
      train/
      val/
      test/
```

The dataset contains two classes:

* Normal
* Pneumonia

---

## Installation

Clone the repository:

```
git clone https://github.com/Rohan-14-1/Pneumonia-Detection-from-Chest-X-ray-Images.git
cd pneumonia-detection-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training the Model

Run the following command to train the CNN model:

```
python main.py
```

This will:

* Load the dataset
* Train the CNN model
* Generate training graphs
* Save model weights

---

## Run The Model

Start the Flask backend server:

```
python api.py
```

API endpoint:

```
/predict
```

---

## Run the Frontend

```
You will get a hosted link after running api.py.
You can also access it locally at:

👉 http://127.0.0.1:5000/

or
👉 http://localhost:5000/
```
Open the following file in your browser and Past local hosted link:

Upload a chest X-ray image to get prediction results.

---

## Results

After training, the following graphs are generated:

```
results/
   loss_plot.png
   accuracy_plot.png
```

These graphs show:

* Training Loss vs Epoch
* Training Accuracy vs Epoch

---

## Model Performance

Evaluation Results:

```
Accuracy: 95.03%
Precision: 94%
Recall: 96%
F1 Score: 95%
```

The high **recall score** indicates that the model successfully detects most pneumonia cases.

---

## Technologies Used

* Python
* PyTorch
* Flask
* HTML
* CSS
* JavaScript
* Deep Learning (CNN)

---

## Author

Rohan Kumar Mandal
<br>
B.Tech Artificial Intelligence and Machine Learning.
<br>
Jain Deemed-to-be University.

