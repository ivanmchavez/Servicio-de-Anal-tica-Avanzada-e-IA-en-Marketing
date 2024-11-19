import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from collections import Counter

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    """
    Plots a confusion matrix.

    Args:
    - cm: Confusion matrix (2D numpy array).
    - target_names: List of labels for the x and y axis.
    - cmap: Colormap to use for the matrix.
    - normalize: Boolean to normalize the matrix.
    - labels: Boolean to display the matrix values.
    - title: Title of the plot.

    Returns:
    - None. Displays the plot.
    """
    # Calculate accuracy and misclassification rate
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # Set colormap if not provided
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # Normalize confusion matrix if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set threshold for text color in cells
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    # Set target names for x and y axis
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    # Display the matrix values
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    # Labels, title and ticks
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def model_predict(model, hist, y_shuffle_train, y_shuffle_valid, y_shuffle_test,
                  image_shuffle_test, text_shuffle_test, y_vrank_shuffle_test, y_lrank_shuffle_test):
    """
    Predicts using the model and plots training/validation performance metrics.

    Args:
    - model: Trained Keras model.
    - hist: History object from model training.
    - y_shuffle_train: Training labels.
    - y_shuffle_valid: Validation labels.
    - y_shuffle_test: Test labels.
    - image_shuffle_test: Test image data.
    - text_shuffle_test: Test text data.
    - y_vrank_shuffle_test: Test video ranking data.
    - y_lrank_shuffle_test: Test like ranking data.

    Returns:
    - None. Prints evaluation metrics and plots performance.
    """
    # Plot training and validation accuracy and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(hist.history['T1_accuracy'])
    ax[0].plot(hist.history['val_T1_accuracy'])
    ax[0].set_title('Popularity (binary) Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['Train', 'Valid'], loc='upper left')

    ax[1].plot(hist.history['T1_loss'])
    ax[1].plot(hist.history['val_T1_loss'])
    ax[1].set_title('Popularity (binary) Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend(['Train', 'Valid'], loc='upper left')
    
    plt.show()

    # Predict using the model
    pred = model.predict([image_shuffle_test, text_shuffle_test, y_vrank_shuffle_test, y_lrank_shuffle_test])

    # Print label distribution in datasets
    print(f"Train label distribution) non-popular: {Counter(y_shuffle_train)[0]}, popular: {Counter(y_shuffle_train)[1]}")
    print(f"Valid label distribution) non-popular: {sum(y_shuffle_valid == 0)}, popular: {sum(y_shuffle_valid == 1)}")
    print(f"Test label distribution) non-popular: {sum(y_shuffle_test == 0)}, popular: {sum(y_shuffle_test == 1)}\n")

    # Generate confusion matrix
    pred_l = np.where(pred[0] > 0.5, 1, 0)
    cm = confusion_matrix(y_shuffle_test, pred_l)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix: \n", cm)
    print(classification_report(y_shuffle_test, pred_l, target_names=['non-popular', 'popular'], digits=3))

    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names=['non-popular', 'popular'], title='Confusion Matrix')

    # Calculate G-Mean score
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    g_mean = math.sqrt(recall * specificity)
    print(f"G-Mean score: {g_mean:.4f}")

    # Print imbalanced classification metrics
    print("Imbalanced metrics: ")
    print(classification_report_imbalanced(y_shuffle_test, pred_l, target_names=['non-popular', 'popular'], digits=3))
