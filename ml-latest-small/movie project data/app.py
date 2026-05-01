from flask import Flask, request, jsonify, render_template
from model_1 import recommend
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for smooth API calls

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        data = request.get_json()
        movie_title = data.get("title", "")
        result = recommend(movie_title, top_n=10)
        
        if result is None or result.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Convert to JSON-friendly format
        response_data = result.to_dict(orient="records")
        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
