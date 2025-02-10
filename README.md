# **🥤 Beverage Price Prediction App**

This is a **machine learning-powered web application** built with **Streamlit** to predict the total price of beverage orders based on various factors such as customer type, product, region, and order details. The model is trained using historical beverage sales data.

## **📌 Features**
✅ Predict beverage order prices based on multiple input parameters  
✅ User-friendly interface built with Streamlit  
✅ Target encoding for categorical variables  
✅ Supports different customer types, products, and regions  

## **📂 Repository Structure**
- **`model.pkl`** - Trained ML model for beverage price prediction
- **`scaler.pkl`** - Saved scaler for numerical feature transformation
- **`app.py`** - Streamlit web app for user interaction and predictions
- **`encoder_product.pkl`** - Encoded mapping for product names  
- **`encoder_region.pkl`** - Encoded mapping for region names  
- **`app.py`** - Streamlit web app for user interaction and predictions  
- **`requirements.txt`** - List of dependencies required to run the project  
- **`README.md`** - Project overview and usage instructions  

## **📊 Dataset**
The dataset used for training the model is available on Kaggle:  
🔗 [Beverage Sales Dataset](https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales)  

## **🚀 How to Run the App**
### **1️⃣ Clone the repository**
```bash
git clone https://github.com/sythe95/BeveragePricePrediction.git
cd BeveragePricePrediction
```
### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```
### **3️⃣ Run the Streamlit app**
```bash
streamlit run app.py
```

## **🖥 Demo**

![Screenshot (156)](https://github.com/user-attachments/assets/357247cf-6702-4c73-a088-aec91cd61997)


![Screenshot (157)](https://github.com/user-attachments/assets/d64a3e4f-a563-48fc-b73e-41de8ec1c612)


![Screenshot (158)](https://github.com/user-attachments/assets/d2a99139-c785-4a95-9e0e-3fb4cb9fef82)


## **📢 Connect & Contribute**
If you have suggestions, feel free to open an issue or contribute via pull requests! 🚀
