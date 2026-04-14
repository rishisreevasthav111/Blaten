from flask import Flask, request, jsonify, render_template
from model_1 import recommend

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    movie_title = data.get("title", "")
    result = recommend(movie_title, top_n=10)
    if result is None:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(result.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)