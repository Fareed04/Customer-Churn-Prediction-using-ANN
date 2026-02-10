# 📉 Customer Churn Prediction — Applied Machine Learning Case Study

## 1️⃣ Business Problem

Customer churn represents a major source of revenue loss for retail banks. Many customers disengage gradually, making churn difficult to detect without data-driven signals.

The objective of this project is to **predict the likelihood of customer churn** using historical customer data, enabling earlier intervention and more targeted retention strategies.

**Target Variable**: `Exited`  
(1 = Customer churned, 0 = Customer retained)

---

## 2️⃣ Dataset & Context

The dataset (`Churn_Modelling.csv`) contains **10,000 customer records** from a retail banking environment, combining demographic attributes with account and engagement behavior.

**Key Features Include**:
- Credit score  
- Geography (France, Germany, Spain)  
- Gender  
- Age and tenure  
- Account balance and estimated salary  
- Product usage and activity status  

The dataset reflects a **realistic churn scenario** with mild class imbalance, typical of financial services use cases.

---

## 3️⃣ Data Preparation & Feature Engineering

To ensure consistency between training and deployment, a structured preprocessing pipeline was applied:

- Removed non-informative identifiers (`RowNumber`, `CustomerId`, `Surname`)
- Label encoding for binary categorical variables (Gender)
- One-hot encoding for multi-class categorical variables (Geography)
- Standard scaling of numerical features
- 70/30 train–test split

All preprocessing artifacts (encoders and scaler) were serialized and reused during inference to avoid training–serving skew.

---

## 4️⃣ Modeling Approach

An **Artificial Neural Network (ANN)** was used to capture non-linear relationships between customer attributes and churn behavior after preprocessing.

### Model Architecture
- Input layer: 12 standardized features  
- Hidden layer 1: 64 neurons (ReLU)  
- Hidden layer 2: 32 neurons (ReLU)  
- Output layer: 1 neuron (Sigmoid)  

**Optimizer**: Adam  
**Loss Function**: Binary Crossentropy  
**Evaluation Metric**: Accuracy  

Early stopping and TensorBoard monitoring were applied to improve generalization and track training behavior.

---

## 5️⃣ Model Performance & Observations

- Validation accuracy stabilized around **86–87%**
- Training and validation loss converged smoothly
- Higher churn risk was commonly associated with:
  - Low customer activity (`IsActiveMember = 0`)
  - Fewer products held
  - High balances without corresponding engagement  

The model is best suited as a **risk-ranking tool**, rather than a standalone decision system.

---

## 6️⃣ Business Value

This solution supports business decision-making by:

- Identifying high-risk customers earlier
- Enabling more targeted retention efforts
- Supporting probability-based prioritization instead of binary rules
- Reducing churn-related revenue leakage

Predictions are expressed as **churn probabilities**, allowing flexible thresholds based on business risk tolerance.

---

## 7️⃣ Deployment: Streamlit Application

A lightweight **Streamlit application** was developed to make the model usable outside a notebook environment.

### Application Features
- Simple customer data input interface
- Automated preprocessing (encoding + scaling)
- Real-time churn probability prediction
- Clear churn vs non-churn output

This demonstrates the transition from model development to a usable analytical tool.

---

## 8️⃣ Project Structure

```

Customer-Churn-Prediction-using-ANN/
│
├── Churn_Modelling.csv             # Raw dataset
├── LICENSE                         # Project license
├── README.md                       # Project documentation
├── app.py                          # Streamlit inference application
├── experiments.ipynb               # Data preprocessing and ANN training workflow
├── prediction.ipynb                # Inference testing notebook
├── model.h5                        # Trained ANN model
├── scaler.pkl                      # StandardScaler used during training
├── label_encoder_gender.pkl        # Gender label encoder
├── one_hot_encoder_geography.pkl   # Geography one-hot encoder
├── requirements.txt                # Project dependencies

````

---

## 9️⃣ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

## 🔟 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* TensorBoard

---

## 11️⃣ Key Takeaways

* Neural networks can effectively model churn-related behavior patterns
* Consistent preprocessing is critical for deployment-ready ML systems
* Probability-based outputs are more actionable than hard classifications

---

## 12️⃣ Potential Improvements

* Address class imbalance using class weights or resampling
* Hyperparameter tuning (e.g., Keras Tuner)
* Model explainability with SHAP or LIME
* Migration to native `.keras` model format
* Cloud deployment (Streamlit Cloud, Render)

---

## 👤 Author

**Fareed**
Data Analyst | Python Developer

---

## 📜 License

For educational and portfolio demonstration purposes.
Tell me which project repo is next.
```
