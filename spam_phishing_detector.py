import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
def load_data():
    # Load the dataset from the CSV file
    data = pd.read_csv('combined_data.csv')
    return data

# Preprocess the data
def preprocess_data(data):
    # Convert labels to binary (if not already)
    data['label'] = data['label'].apply(lambda x: 1 if x == 1 else 0)
    
    # Separate features and labels
    X = data['text']  # Message content
    y = data['label']  # Labels (1 = spam, 0 = not spam)
    
    return X, y

# Train the model
def train_model(X_train, y_train):
    # Vectorize the text data (convert text to numeric features using TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

# Save the model and vectorizer
def save_model(model, vectorizer):
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Predict using the trained model
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate and display metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main function
if __name__ == "__main__":
    # Step 1: Load the data
    data = load_data()
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(data)
    
    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Train the model
    model, vectorizer = train_model(X_train, y_train)
    
    # Step 5: Save the model and vectorizer
    save_model(model, vectorizer)
    
    # Step 6: Evaluate the model
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # Step 7: Test the model with new inputs
    while True:
        message = input("\nEnter a message to check (or type 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        # Transform the message using the vectorizer
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)
        result = "Spam" if prediction == 1 else "Not Spam"
        print(f"The message is classified as: {result}")
