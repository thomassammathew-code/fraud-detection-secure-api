from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import datetime
import os


app = Flask(__name__)


limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute"]
)


model = joblib.load("model/fraud_model.pkl")


API_KEY = "mysecurekey123"


LOG_FILE = "logs/activity.log"
BLOCKED_IP_FILE = "blocked_ips.txt"

if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists(BLOCKED_IP_FILE):
    open(BLOCKED_IP_FILE, "w").close()


def log_event(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")


def is_blocked(ip):
    with open(BLOCKED_IP_FILE, "r") as f:
        blocked_ips = f.read().splitlines()
    return ip in blocked_ips

def block_ip(ip):
    with open(BLOCKED_IP_FILE, "a") as f:
        f.write(ip + "\n")
    log_event(f"BLOCKED IP: {ip}")


request_counts = {}

def track_requests(ip):
    now = datetime.datetime.now()

    if ip not in request_counts:
        request_counts[ip] = []

    request_counts[ip].append(now)

    request_counts[ip] = [
        t for t in request_counts[ip]
        if (now - t).seconds < 60
    ]

    if len(request_counts[ip]) > 10:
        block_ip(ip)


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": []})

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    return jsonify({"logs": lines[-30:]})


@app.route("/predict", methods=["POST"])
@limiter.limit("100 per minute")
def predict():
    try:
        ip = request.remote_addr

        if is_blocked(ip):
            log_event(f"Blocked IP tried access: {ip}")
            return jsonify({"error": "Your IP is blocked"}), 403

        track_requests(ip)

        user_api_key = request.headers.get("x-api-key")
        if user_api_key != API_KEY:
            log_event(f"Unauthorized access from {ip}")
            return jsonify({"error": "Unauthorized"}), 401

        data = request.json

        if not data or "features" not in data:
            return jsonify({"error": "Invalid input"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            "fraud": bool(prediction),
            "risk_score": float(probability)
        }

        if probability > 0.8:
            log_event(f"HIGH RISK from {ip} - Score: {probability}")

        return jsonify(result)

    except Exception as e:
        log_event(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
