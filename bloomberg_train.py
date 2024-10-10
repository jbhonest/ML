import joblib
from sqlalchemy import create_engine
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    articles['content'], articles['category'], test_size=0.2, random_state=42)

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

joblib.dump(model, 'article_classifier.pkl')
