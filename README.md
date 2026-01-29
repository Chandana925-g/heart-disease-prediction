# Heart Disease Prediction System 🫀

A machine learning web app built with Streamlit to predict heart disease risk using Logistic Regression and Random Forest models. This application provides an interactive interface for doctors and users to input health parameters and get instant predictions along with probability scores.

## 🚀 Features
*   **Instant Predictions:** Predicts heart disease risk using a trained **Random Forest Classifier** (and Logistic Regression for comparison).
*   **Interactive UI:** User-friendly sidebar with sliders and dropdowns for easy data entry.
*   **Visual Analytics:** 
    *   **Correlation Heatmap:** To understand relationships between different health factors.
    *   **Histograms:** To visualize data distributions (e.g., Age vs. Disease).
*   **Model Comparison:** clearly shows the accuracy difference between multiple algorithms.

## 🛠️ Tech Stack
*   **Python**
*   **Streamlit** (Frontend)
*   **Scikit-Learn** (Machine Learning)
*   **Pandas & NumPy** (Data Manipulation)
*   **Seaborn & Matplotlib** (Visualization)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

1.  Navigate to the project directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  The app will open automatically in your web browser at `http://localhost:8501`.

## 📊 Dataset
The dataset used is the famous **Cleveland Heart Disease Dataset** (often found on Kaggle/UCI ML Repository). It contains 13 clinical parameters:
*   **Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps), Cholesterol (chol), etc.**

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements (e.g., adding more models like SVM or Neural Networks), feel free to fork the repo and submit a Pull Request.

## 📜 License
This project is open-source.
