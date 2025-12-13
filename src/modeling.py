import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight
from preprocessing import preprocessed_data


X_train, X_test, y_train, y_test, df = preprocessed_data(save_path=None)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = (y_train - y_train.min()).astype(int)
y_test = (y_test - y_test.min()).astype(int)

num_classes = len(np.unique(y_train))
input_dim = X_train.shape[1]

# Building Classification Model
def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

model = build_model(input_dim, num_classes)

# Early Stopping Implementation
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

classes = np.unique(y_train)
weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=classes, y=y_train
)
class_weights = dict(zip(classes, weights))

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights,
    verbose=2
)

# Model Evaluation
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save and Loading
model.save("model/final_model.keras")

loaded_model = tf.keras.models.load_model("model/final_model.keras")

# Test Set Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

# Deep Model Evaluation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

f1_scores = f1_score(y_test, y_pred, average=None)
sns.barplot(x=np.unique(y_test), y=f1_scores)
plt.title("F1 Score per Class")
plt.xlabel("Severity")
plt.ylabel("F1 Score")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
