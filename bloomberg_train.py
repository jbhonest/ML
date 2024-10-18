import joblib
from sqlalchemy import create_engine
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


host = "localhost"
dbname = "ml"
user = "ml_user"
password = "ml_10925"
port = 5432


# Create the SQLAlchemy engine
engine = create_engine(
    f'postgresql://{user}:{password}@{host}:{port}/{dbname}')


# Query the articles table
query = "SELECT content, category FROM articles"
articles = pd.read_sql(query, engine)

X = articles['content']
y = articles['category']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create a pipeline for TF-IDF and Logistic Regression
model = make_pipeline(TfidfVectorizer(
    stop_words='english'), LogisticRegression())

# Train the model
model.fit(X_train, y_train)


# Access the TfidfVectorizer and classifier
vectorizer = model.named_steps['tfidfvectorizer']
classifier = model.named_steps['logisticregression']

terms = vectorizer.get_feature_names_out()
coefficients = classifier.coef_
categories = classifier.classes_
category_top_terms = {}

# Define how many top terms to extract
top_n = 10

# Loop through each category
for i, category in enumerate(categories):
    # Get the coefficients for the current category
    category_coefficients = coefficients[i]

    # Get the indices of the top n terms for the category (highest coefficients)
    top_term_indices = np.argsort(category_coefficients)[-top_n:][::-1]

    # Get the top terms and their corresponding coefficients
    top_terms = [(terms[j], category_coefficients[j])
                 for j in top_term_indices]

    # Store the top terms for this category
    category_top_terms[category] = top_terms

# Convert the result to a DataFrame for easier Excel writing
result_data = []

for category, terms in category_top_terms.items():
    for term, score in terms:
        result_data.append({
            'Category': category,
            'Term': term,
            'Coefficient': score
        })

# Create a DataFrame from the result
result_df = pd.DataFrame(result_data)

# Write the DataFrame to an Excel file
output_path = 'top_terms_by_category.xlsx'
result_df.to_excel(output_path, index=False)
print(f"Top terms for each category have been written to {output_path}")

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='macro')

# Print the F1 score
print(f"F1 Score: {f1:.2f}")

# Save the model
classifier_file = 'article_classifier.pkl'
joblib.dump(model, classifier_file)
print(f"The model has been written to {classifier_file}")


# Plot the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=model.classes_, cmap='Blues')
plt.title('Confusion Matrix')


# Define the classes
classes = ["technology", "economics", "politics"]

# Get predicted probabilities (needed for ROC curves)
y_score = model.predict_proba(X_test)

# Binarize the test labels (needed for ROC curve and AUC)
y_test_bin = label_binarize(y_test, classes=classes)

# Plotting the ROC curves for each class
plt.figure(figsize=(8, 6))


for i, class_name in enumerate(classes):

    # Compute ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve ({
             class_name}, AUC = {roc_auc:.2f})')

# Plot the random guess line (diagonal)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Set plot labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves')
plt.legend(loc="lower right")

# Show the plot
plt.show()
