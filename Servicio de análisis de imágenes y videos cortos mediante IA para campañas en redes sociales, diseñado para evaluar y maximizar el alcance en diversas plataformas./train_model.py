# train_model.py
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from multimodal_model import MultimodalModel
from data_loader import prepare_data

# Cargar y preparar los datos
(X_img_train, X_text_train, X_meta_train, y_train), (X_img_val, X_text_val, X_meta_val, y_val) = prepare_data('/mnt/data/instagram_data.csv')

# Crear el modelo
model = MultimodalModel()

# Callbacks para guardar el mejor modelo y detener el entrenamiento temprano
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Entrenamiento del modelo
history = model.fit(
    [X_img_train, X_text_train, X_meta_train],
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=([X_img_val, X_text_val, X_meta_val], y_val),
    callbacks=[checkpoint, early_stopping]
)

# Guardar las variables de validación para evaluación posterior
np.savez('validation_data.npz', X_img_val=X_img_val, X_text_val=X_text_val, X_meta_val=X_meta_val, y_val=y_val)
