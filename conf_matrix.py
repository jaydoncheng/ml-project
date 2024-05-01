import numpy as np
from keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = np.load('samples/word_vectors-20k-300-300.npy')
y = np.load('samples/categories_words-20k-300-300.npy')
def words_to_categories(words):
    if words == 'positive':
        return [1, 0, 0]
    elif words == 'negative':
        return [0, 0, 1]
    elif words == 'neutral':
        return [0, 1, 0]

y = np.array([words_to_categories(i) for i in y])

print('loaded data')

print('splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cnn_model = load_model('models/cnn_model.keras')

loss, accuracy = cnn_model.evaluate(X_test, y_test, verbose=2)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % loss)

y_pred_prob = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_pred_classes = np.array([x - 1 for x in y_pred_classes])

# convert multiclass probabilities to binary

print(y_test)
print(f"y_pred_prob: {y_pred_prob}")
print(f"y_pred_prob: {y_pred_classes}")

# convert y_test to -1 to 1
y_test = np.argmax(y_test, axis=1)
y_test = np.array([x - 1 for x in y_test])

conf_matrix = confusion_matrix(y_test, y_pred_classes)

class_names = ['positive', 'neutral', 'negative']
# calculate precision, recall, f1-score
true_positives = np.diag(conf_matrix)
false_pos = np.sum(conf_matrix, axis=0) - true_positives
false_neg = np.sum(conf_matrix, axis=1) - true_positives

precision = true_positives / (true_positives + false_pos)
recall = true_positives / (true_positives + false_neg)
f1 = 2 * precision * recall / (precision + recall)
print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}')

# display a precision recall f1 table
from tabulate import tabulate

#round to 3 decimal places
precision = np.round(precision, 3)
recall = np.round(recall, 3)
f1 = np.round(f1, 3)

print(tabulate([['positive', precision[0], recall[0], f1[0]],
                ['neutral', precision[1], recall[1], f1[1]],
                ['negative', precision[2], recall[2], f1[2]]],
                headers=['class', 'precision', 'recall', 'f1']))


print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot()
plt.savefig('conf_matrix.png')
