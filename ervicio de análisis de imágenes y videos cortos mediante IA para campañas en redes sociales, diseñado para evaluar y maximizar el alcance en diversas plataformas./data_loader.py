# data_loader.py
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ruta al archivo de datos
data_path = '/mnt/data/instagram_data.csv'

# Cargar datos
def load_data(data_path):
    df = pd.read_csv(data_path)
    print(df.head())  # Ver los primeros registros para entender la estructura
    return df

# Preprocesamiento de imágenes
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0  # Normalizar
        return image
    except Exception as e:
        print(f"Error al cargar la imagen {image_path}: {e}")
        return np.zeros((224, 224, 3))  # Placeholder para imágenes no cargadas

# Preprocesamiento de texto con BERT
def preprocess_text(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='tf')
    return tokenized_texts['input_ids']

# Preprocesamiento de metadatos
def preprocess_metadata(metadata_df):
    # Normalizar metadatos 
    return metadata_df.to_numpy()

# Función principal para cargar y preprocesar datos
def prepare_data(data_path):
    df = load_data(data_path)
    
    # Extraer y preprocesar características
    X_images = np.array([preprocess_image(img_path) for img_path in df['image_path']])  # Ajusta el nombre de columna
    X_texts = preprocess_text(df['description'])  # Ajusta el nombre de columna
    X_metadata = preprocess_metadata(df[['likes', 'comments', 'shares']])  # Ajusta nombres de columnas
    y = df['is_popular'].values  # Etiqueta de clasificación; ajustar según tu CSV
    
    # Dividir en conjuntos de entrenamiento y validación
    X_img_train, X_img_val, X_text_train, X_text_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        X_images, X_texts, X_metadata, y, test_size=0.2, random_state=42)
    
    return (X_img_train, X_text_train, X_meta_train, y_train), (X_img_val, X_text_val, X_meta_val, y_val)
