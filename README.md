# 📉 Customer Churn Prediction using ANN — Case Study

## 1️⃣ Business Problem

Retail banks lose significant revenue when customers churn, often without clear early warning signals. The objective of this project is to **predict whether a customer is likely to exit the bank** using historical customer data, enabling data-driven retention strategies.

**Key Question**: *Can we identify high-risk customers before churn occurs using behavioral and demographic data?*
**Target Variable**: `Exited` (1 = Churn, 0 = Retained)

---

## 2️⃣ Dataset & Context

The dataset (`Churn_Modelling.csv`) contains **10,000 customer records** from a retail bank, covering demographic attributes, account behavior, and engagement indicators.

**Core Features**:

* CreditScore
* Geography (France, Germany, Spain)
* Gender
* Age, Tenure
* Balance, EstimatedSalary
* Product usage and activity status

This dataset reflects a **realistic, imbalanced churn scenario** commonly seen in financial services.

---

## 3️⃣ Data Preparation & Feature Engineering

To ensure model reliability and deployment consistency, the following preprocessing pipeline was applied:

* Dropped non-informative identifiers (`RowNumber`, `CustomerId`, `Surname`)
* **Label Encoding** for Gender
* **One-Hot Encoding** for Geography
* **Standard Scaling** for numerical features
* 70/30 Train–Test split

All transformers (encoders and scaler) were persisted using Pickle and reused during inference in the Streamlit app.

---

## 4️⃣ Modeling Approach

An **Artificial Neural Network (ANN)** was selected to capture non-linear relationships between customer attributes and churn behavior.

### Model Architecture

* Input Layer: 12 standardized features
* Hidden Layer 1: 64 neurons (ReLU)
* Hidden Layer 2: 32 neurons (ReLU)
* Output Layer: 1 neuron (Sigmoid)

**Optimizer**: Adam (learning rate = 0.01)
**Loss Function**: Binary Crossentropy
**Metric**: Accuracy

Early Stopping and TensorBoard monitoring were used to prevent overfitting and track convergence.

---

## 5️⃣ Model Performance & Insights

* **Validation Accuracy**: ~86–87%
* The model converged quickly with stable validation loss
* Customers with higher churn likelihood often showed:

  * Lower engagement (`IsActiveMember = 0`)
  * Fewer products
  * Higher balances without corresponding activity

While accuracy is strong, this model is best positioned as a **risk-ranking tool**, not a final decision-maker.

---

## 6️⃣ Business Impact

This solution enables the business to:

* Proactively flag high-risk customers
* Prioritize retention campaigns more effectively
* Reduce churn-related revenue loss
* Support decision-making with probability-based outputs

The output is a **churn probability**, allowing flexible threshold selection depending on business risk tolerance.

---

## 7️⃣ Deployment: Streamlit Application

A lightweight **Streamlit web application** was built to operationalize the model.

### App Capabilities

* User-friendly customer data input
* Automated encoding and scaling
* Real-time churn probability prediction
* Clear churn vs non-churn classification

This bridges the gap between experimentation and real-world usability.

---

## 8️⃣ Project Structure

```
Customer-Churn-Prediction-using-ANN/
│
├── Churn_Modelling.csv             # Raw dataset
├── LICENSE                         # Project license
├── README.md                       # Project documentation
├── app.py                          # Streamlit application for inference
├── experiments.ipynb               # Data preprocessing & ANN training workflow
├── prediction.ipynb                # Standalone notebook for churn prediction testing
├── model.h5                        # Trained ANN model
├── scaler.pkl                      # StandardScaler used during training
├── label_encoder_gender.pkl        # Gender LabelEncoder
├── one_hot_encoder_geography.pkl   # Geography OneHotEncoder
├── requirements.txt                # Project dependencies
```

---

## 9️⃣ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

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

* ANN models can effectively capture churn behavior patterns
* Deployment-ready ML requires consistent preprocessing pipelines
* Probability-based predictions are more actionable than binary labels

---

## 12️⃣ Future Improvements

* Address class imbalance using class weights or SMOTE
* Hyperparameter tuning (Keras Tuner)
* Model explainability with SHAP or LIME
* Replace legacy `.h5` model with native `.keras` format
* Cloud deployment (Streamlit Cloud / Render)

---

## 👤 Author

**Fareed**
Data Analyst • Applied Data Scientist • Python Developer

---

## 📜 License

Educational and portfolio use only.
