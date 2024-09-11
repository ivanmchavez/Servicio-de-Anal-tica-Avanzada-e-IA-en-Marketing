# generate_recommendations.py
def generate_recommendations(y_pred_classes, X_img_val, X_text_val, X_meta_val):
    for i, prediction in enumerate(y_pred_classes):
        if prediction == 0:
            print(f"Post {i}: Considera mejorar la imagen ajustando la iluminación o utilizando filtros más llamativos.")
            print(f"Descripción: {X_text_val[i]} - Podrías incluir hashtags más relevantes.")
        else:
            print(f"Post {i}: Mantén la estrategia actual. La proyección de popularidad es buena.")
