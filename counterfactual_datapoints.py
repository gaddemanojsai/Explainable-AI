import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# for i in range(len(y)):
#   print(y[i])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Select a single instance for testing
test_instance = X_test[17].reshape(1, -1)

# Predict the class label for the test instance
predicted_label = model.predict(test_instance)

output_counterfactuals = []

# Generate counterfactual points for the test instance
def generate_counterfactual_points(model, data, output, epsilon=0.1, num_samples=100):
    counterfactuals = []

    for original_data, original_output in zip(data, output):
        modified_data = np.copy(original_data)  # Make a copy of the original data point

        # Generate multiple counterfactual samples for each data point
        while(True):
            if(len(counterfactuals)==100):
              break
            perturbation = np.random.uniform(low=-epsilon, high=epsilon, size=original_data.shape)
            modified_data = np.round(modified_data + perturbation, decimals=1)
            if all(x > 0 for x in modified_data):
              # Make a prediction with the modified data
              predicted_output = model.predict(modified_data.reshape(1, -1))



              # Check if the prediction is different from the original output
              if predicted_output != original_output:
                  counterfactuals.append(modified_data)
                  output_counterfactuals.append(predicted_output)
            # Reset the modified data to the original point for the next iteration
            modified_data = np.copy(original_data)

    return counterfactuals

# Generate counterfactual points for the test instance
counterfactuals = generate_counterfactual_points(model, test_instance, predicted_label)

print("test_instance",test_instance)
print("predicted_label",predicted_label)

# Print the generated counterfactual points
for counterfactual in counterfactuals:
    print(counterfactual)

for temp in output_counterfactuals:
    print(temp,end=' ')