import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load data
data_path = "data/emotions.csv"  # Replace with actual dataset path
df = pd.read_csv(data_path)

# Preprocessing
X = np.array(df['pixels'].tolist(), dtype="float32") / 255.0
X = X.reshape(-1, 48, 48, 1)  # assuming 48x48 grayscale images
y = pd.get_dummies(df['emotion']).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 emotions
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("face_emotionModel.h5")
print("Model saved!")
