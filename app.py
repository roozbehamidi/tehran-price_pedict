import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# تنظیمات اولیه
# -----------------------------

# لیست آدرس‌ها
from utils import address_list  # مطمئن شو utils.address_list حاوی 192 آدرس است

# بارگذاری مدل
model = joblib.load('final_xgb_model.joblib')

# ستون‌های ورودی مدل: ویژگی‌های عددی + dummy آدرس‌ها به ترتیب الفبایی
columns = ["Area", "Room", "Parking", "Warehouse", "Elevator"] + sorted(address_list)

st.title('House Price Prediction in Tehran')

# -----------------------------
# دریافت ورودی‌ها از کاربر
# -----------------------------
area = st.number_input("Area (متر مربع)", 0, 10000, step=1, format="%d")

room = st.number_input("Room (تعداد اتاق)", 0, 100, step=1, format="%d")

parking = st.radio("Parking:", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
warehouse = st.radio("Warehouse:", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
elevator = st.radio("Elevator:", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

address = st.selectbox("Address:", address_list)

# -----------------------------
# تابع تبدیل آدرس به one-hot
# -----------------------------
def encode_address(address_value, columns):
    """
    ورودی:
        address_value: آدرس انتخابی کاربر
        columns: لیست ستون‌های ورودی مدل (بدون ستون Price)
    خروجی:
        دیکشنری {آدرس: مقدار} با 1 برای آدرس انتخابی و بقیه صفر
    """
    # فقط ستون‌های آدرس را پیدا کن و مرتب کن
    address_cols = sorted([c for c in columns if c in address_list])
    encoded = {col: 0 for col in address_cols}
    if address_value in encoded:
        encoded[address_value] = 1
    return encoded

# -----------------------------
# تابع پیش‌بینی
# -----------------------------
def predict(): 
    # ویژگی‌های عددی/باینری
    row = {
        "Area": area,
        "Room": room,
        "Parking": parking,
        "Warehouse": warehouse,
        "Elevator": elevator,
    }

    # دیتافریم نهایی با صفر
    X = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # مقداردهی به ویژگی‌های عددی/باینری
    for col, val in row.items():
        if col in X.columns:
            X.at[0, col] = val

    # مقداردهی آدرس
    address_encoding = encode_address(address, columns)
    for col, val in address_encoding.items():
        if col in X.columns:
            X.at[0, col] = val

    # پیش‌بینی
    prediction = model.predict(X)
    st.write("Predicted Price:", f"{prediction[0]:,.0f} ریال")

# -----------------------------
# دکمه پیش‌بینی
# -----------------------------
trigger = st.button('Predict', on_click=predict)
