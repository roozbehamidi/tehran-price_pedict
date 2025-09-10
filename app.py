import streamlit as st
from utils import encode_address, columns, address_list

import numpy as np
import pandas as pd
import joblib

# بارگذاری مدل
model = joblib.load('final_xgb_model.joblib')

st.title('House Price Prediction in Tehran')

# دریافت ورودی‌ها از کاربر
area = st.number_input("Area", 0, 10000)

room = st.radio(
    "Room :",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
parking = st.radio(
    "Parking :",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
warehouse = st.radio(
    "Warehouse :",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
elevator = st.radio(
    "Elevator :",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

address = st.selectbox(
    "Address :",
    address_list
)




# -----------------------------
# تابع پیش‌بینی
# -----------------------------
def predict(): 
    # ویژگی‌های اصلی
    row = {
        "Area": area,
        "Room": room,
        "Parking": parking,
        "Warehouse": warehouse,
        "Elevator": elevator,
    }

    # ساخت دیتافریم نهایی با همه ستون‌ها (صفر پر می‌کنیم)
    X = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # مقداردهی به ویژگی‌های عددی/باینری
    for col, val in row.items():
        if col in X.columns:
            X.at[0, col] = val

    # مقداردهی به آدرس
    address_encoding = encode_address(address, columns)
    for col, val in address_encoding.items():
        if col in X.columns:
            X.at[0, col] = val

    # پیش‌بینی
    prediction = model.predict(X)
    st.write("Predicted Price:", f"{prediction[0]:,.0f} ریال")


# دکمه پیش‌بینی
trigger = st.button('Predict', on_click=predict)

