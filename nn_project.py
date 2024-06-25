# Импорт библиотек
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms as T
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import streamlit as st
from PIL import Image
from models.model import MyModel
from models.preprocessing import preprocess


@st.cache_resource()
# Функция загрузки модели из файла weights.pt
def load_model():
    model = (
        MyModel()
    )  # Загрузка вашей модели, должна соответствовать MyModel в model.py
    model.load_state_dict(
        torch.load("weights.pt", map_location=torch.device("cpu"))
    )  # Загрузка весов
    model.eval()  # Перевод модели в режим eval
    return model


# Функция обработки изображения
def process_image(image):
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)  # Добавление размерности пакета
    return image


# Функция предсказания класса
def predict(image, model):
    outputs = model(image)
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Используем основной выход (logits) из InceptionOutputs
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Интерфейс Streamlit
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Загрузка модели
    model = load_model()

    # Обработка и предсказание
    processed_image = process_image(image)
    prediction = predict(processed_image, model)

    # Отображение результата
    class_names = {
        0: "Building",
        1: "Forest",
        2: "Glacier",
        3: "Mountain",
        4: "Sea",
        5: "Street",
    }  # Ваши классы
    st.write(f"Predicted Class: {class_names[prediction]}")
