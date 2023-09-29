from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/tes', methods=["POST"])
def tes():
   data = request.get_json()
   return jsonify(data), 200

if __name__ == "__main__":
   app.run(debug=True)