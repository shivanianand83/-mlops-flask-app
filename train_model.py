import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Sample data: Weight (lbs) vs Height (inches) for MLB players
# Based on realistic MLB player statistics
# Heights range from 5'10" (70 inches) to 6'6" (78 inches)
# Weights range from 160 lbs to 250 lbs

# Training data (Weight, Height in inches)
# Created a realistic dataset of MLB players
weights = np.array([
    165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250,
    168, 172, 178, 182, 188, 192, 198, 202, 208, 212, 218, 222, 228, 232, 238, 242, 248,
    160, 175, 185, 195, 205, 215, 225, 235, 245,
    172, 182, 192, 202, 212, 222, 232, 242
]).reshape(-1, 1)

heights = np.array([
    70, 71, 72, 72, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80,
    70, 71, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 80,
    69, 72, 73, 74, 75, 76, 77, 78, 79,
    71, 72, 73, 74, 75, 76, 77, 78
])

# Create and train the model
model = LinearRegression()
model.fit(weights, heights)

# Print model information
print("Model trained successfully!")
print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared score: {model.score(weights, heights):.4f}")

# Test the model with some sample weights
test_weights = [180, 200, 220, 250]
for weight in test_weights:
    pred_height = model.predict([[weight]])[0]
    feet = int(pred_height // 12)
    inches = int(pred_height % 12)
    print(f"Weight: {weight} lbs -> Height: {pred_height:.1f} inches ({feet}'{inches}\")")

# Save the model
joblib.dump(model, 'model.joblib')
print("\nModel saved as 'model.joblib'")