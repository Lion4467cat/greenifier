from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    import requests
    import cv2
    import numpy as np

    data = request.get_json()

    lat = data["latitude"]
    lon = data["longitude"]

    # 🔥 Get image from frontend
    image_data = data["image"]

    # Decode base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    with open("sat.png", "wb") as f:
        f.write(image_bytes)

    image = cv2.imread("sat.png")

    if image is None:
        return jsonify({"energy": 0, "cost": 0, "roi": 0})

    # 3️⃣ Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4️⃣ Blur (important)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5️⃣ Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # 6️⃣ Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roof_area = 0

    # 7️⃣ Filter contours
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:  # remove noise
            roof_area += area
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    # Save debug image
    cv2.imwrite("detected_roofs.png", image)

    # 8️⃣ Convert to real area (approx)
    area = roof_area * 0.5

    # 9️⃣ Solar calculation
    system_size_kw = area / 10  # 10 m² per kW

    # Bangalore sunlight factor
    energy = round(system_size_kw * 4.5, 2)

    # 🔟 Cost estimation
    cost = int(system_size_kw * 60000)

    # ROI
    yearly_savings = energy * 365 * 6
    roi = round((yearly_savings / cost) * 100, 2) if cost > 0 else 0

    return jsonify({"energy": energy, "cost": cost, "roi": roi})


if __name__ == "__main__":
    app.run(debug=True)
