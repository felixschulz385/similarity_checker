from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to calculate TF-IDF manually
def compute_tfidf(submissions):
    tokenized = [submission.lower().split() for submission in submissions]
    vocabulary = sorted(set(word for words in tokenized for word in words))
    vocab_index = {word: i for i, word in enumerate(vocabulary)}

    tf = np.zeros((len(submissions), len(vocabulary)))
    for i, words in enumerate(tokenized):
        for word in words:
            tf[i, vocab_index[word]] += 1

    df = np.sum(tf > 0, axis=0)
    idf = np.log(len(submissions) / (df + 1))

    tfidf = tf * idf
    return tfidf

# Function to compute cosine similarity
def compute_cosine_similarity(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / (norm + 1e-9)
    similarity = np.dot(normalized_matrix, normalized_matrix.T)
    return similarity

# Function to calculate similarities
def calculate_similarity(submissions, threshold=0.8):
    tfidf_matrix = compute_tfidf(submissions)
    similarity_matrix = compute_cosine_similarity(tfidf_matrix)

    warnings = []
    num_submissions = len(submissions)
    for i in range(num_submissions):
        for j in range(i + 1, num_submissions):
            if similarity_matrix[i, j] > threshold:
                warnings.append((i, j, similarity_matrix[i, j]))

    return warnings

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle file uploads and process data
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith(".xlsx"):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            df = pd.read_excel(filepath)
            if "Textabgabe" not in df.columns:
                return jsonify({"error": "Excel file must contain a 'Textabgabe' column."}), 400

            submissions = df["Textabgabe"].dropna().tolist()
            threshold = 0.5
            warnings = calculate_similarity(submissions, threshold)

            if warnings:
                results = [
                    f"Submissions {i} and {j} have a similarity score of {score:.2f}"
                    for i, j, score in warnings
                ]
            else:
                results = ["No similarities above the threshold were detected."]

            return render_template("results.html", results=results)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Please upload an Excel file."}), 400

if __name__ == "__main__":
    app.run(debug=True)
