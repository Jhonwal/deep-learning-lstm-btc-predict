# Bitcoin Price Prediction Project

A full-stack web application for predicting Bitcoin prices using deep learning. The project consists of a **React** frontend for user interaction and a **Python** backend for handling data processing, model training, and predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)

---


## Project Overview

This project aims to predict Bitcoin prices using a **Long Short-Term Memory (LSTM)** neural network. The application allows users to:
- Upload historical Bitcoin price data.
- Train the LSTM model on the uploaded data.
- View training results and model performance.
- Make predictions on new data.

The project is divided into two main components:
- **Frontend**: Built with React for a user-friendly interface.
- **Backend**: Built with Python (Flask) for data processing, model training, and serving predictions.

---

## Technologies Used

### Frontend
- **React**: A JavaScript library for building user interfaces.
- **Vite**: A fast build tool for modern web applications.
- **Axios**: For making HTTP requests to the backend API.
- **CSS**: For styling the application.

### Backend
- **Python**: The programming language used for the backend.
- **Flask**: A lightweight web framework for building the API.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For data scaling (e.g., `MinMaxScaler`).

### Data
- Historical Bitcoin price data in CSV format (e.g., `BTCUSDT-12h.csv`, `BTCUSD_1m_Binance.csv`).

---

## Setup and Installation

### Prerequisites
- Node.js and npm (for the frontend)
- Python 3.x (for the backend)
- A package manager like `pip` for Python dependencies

### Backend Setup

1. Navigate to the `backend` folder:
   ```
   cd backend
   ```
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required Python packages:
```
pip install -r requirements.txt
```
4. Run the backend server:
```
python app.py
```

### Frontend setup

1. Navigate to the `frontend` folder:
```
cd frontend
```
2. Install required packages:
```
npm install
```
3. Start the frontend server:
```
npm run dev
```

### Running the Application

1. Start the backend server (follow the steps in Backend Setup).

2. Start the frontend development server (follow the steps in Frontend Setup).

3. Open your browser and navigate to http://localhost:5173 to view the application.