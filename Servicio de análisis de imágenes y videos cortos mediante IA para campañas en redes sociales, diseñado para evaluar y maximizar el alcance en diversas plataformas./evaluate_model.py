# evaluate_model.py
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from plot_metrics import plot_confusion_matrix  # Asegúrate de que este archivo esté presente
from multimodal_model import MultimodalModel

# Cargar el modelo pre-entrenado
model = MultimodalModel()
model.load_weights('best_model.h5')

# Cargar datos de validación guardados
validation_data = np.load('validation_data.npz')
X_img_val = validation_data['X_img_val']
X_text_val = validation_data['X_text_val']
X_meta_val = validation_data['X_meta_val']
y_val = validation_data['y_val']

# Evaluación del modelo
def evaluate_model(model, X_img_val, X_text_val, X_meta_val, y_val):
    # Realizar predicciones
    y_pred = model.predict([X_img_val, X_text_val, X_meta_val])
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convertir predicciones a clases binarias

    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred_classes)
    plot_confusion_matrix(cm, target_names=['non-popular', 'popular'])
    
    # Informe de clasificación
    print(classification_report(y_val, y_pred_classes, target_names=['non-popular', 'popular']))

# Ejecutar evaluación
evaluate_model(model, X_img_val, X_text_val, X_meta_val, y_val)
