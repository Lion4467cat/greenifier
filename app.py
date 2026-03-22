import math
import os
from groq import Groq
import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# 1. Satellite image fetcher — free, no API key
# ---------------------------------------------------------------------------


def fetch_satellite_image(lat: float, lon: float, zoom: int = 18) -> np.ndarray:
    """
    Fetches satellite tiles using the same free Google tile URL
    that Leaflet already uses on the frontend. No API key, no account.

    Fetches a 3x3 grid of tiles and stitches them, then crops the
    center 640x640 pixels around the clicked point.
    Each tile is 256x256px → 3x3 grid = 768x768px → crop to 640x640.
    """
    # Convert lat/lon to tile x/y (standard Web Mercator formula)
    lat_r = math.radians(lat)
    n = 2**zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int(
        (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }  # required — bare Python requests get blocked

    # Fetch 3×3 grid of tiles centered on the clicked point
    tiles = []
    for dy in [-1, 0, 1]:
        row = []
        for dx in [-1, 0, 1]:
            tx = tile_x + dx
            ty = tile_y + dy
            url = f"https://mt1.google.com/vt/lyrs=s&x={tx}&y={ty}&z={zoom}"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            tile = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if tile is None:
                raise ValueError(f"Failed to decode tile x={tx} y={ty} z={zoom}")
            row.append(tile)
        tiles.append(row)

    # Stitch rows horizontally, then stack vertically → 768×768 image
    stitched = cv2.vconcat([cv2.hconcat(row) for row in tiles])

    # Crop center 640×640 — focused on the clicked coordinate
    h, w = stitched.shape[:2]
    cx, cy = w // 2, h // 2
    img = stitched[cy - 320 : cy + 320, cx - 320 : cx + 320]

    return img


# ---------------------------------------------------------------------------
# 2. Ground resolution — pixels → real-world meters
# ---------------------------------------------------------------------------


def meters_per_pixel(lat: float, zoom: int = 18) -> float:
    """
    At zoom=18, this is ~0.6 m/px near the equator.
    Adjusts for latitude (pixels shrink as you move toward the poles).
    """
    earth_circumference = 40_075_016.686  # metres
    return (earth_circumference * math.cos(math.radians(lat))) / (256 * (2**zoom))


def get_peak_sun_hours(lat: float, lon: float) -> float:
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?latitude={lat}&longitude={lon}"
        "&parameters=ALLSKY_SFC_SW_DWN"
        "&community=RE"
        "&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        monthly = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        return monthly.get("ANN", 5.2)
    except Exception:
        return 5.2  # fallback if API fails


# ---------------------------------------------------------------------------
# 3. Roof / usable surface detector
# ---------------------------------------------------------------------------


def detect_usable_surfaces(img: np.ndarray) -> dict:
    """
    Detects flat rooftops and open land using color thresholding + morphology.

    Rooftops : concrete / metal  → low saturation, medium-high brightness in HSV
    Open land: bare earth        → warm hue, low-medium saturation

    Returns pixel area and surface count only — no pixel coordinates,
    which avoids the pixel-to-map alignment bug from the old html2canvas approach.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Concrete / metal rooftops: low saturation, bright
    roof_mask = cv2.inRange(hsv, (0, 0, 120), (180, 55, 255))

    # Bare land / open ground: warm brownish tones
    land_mask = cv2.inRange(hsv, (8, 20, 80), (32, 130, 210))

    combined = cv2.bitwise_or(roof_mask, land_mask)

    # Remove road-like linear structures from the mask
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick_edges = cv2.dilate(edges, kernel, iterations=2)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(thick_edges))

    # Morphological cleanup — close small holes, remove speckle
    clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, clean)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean)

    # Find and filter contours by area
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA_PX = 150  # below this = noise
    MAX_AREA_PX = 70_000  # above this = sky / water artifact

    valid = [c for c in contours if MIN_AREA_PX < cv2.contourArea(c) < MAX_AREA_PX]
    total_px = sum(cv2.contourArea(c) for c in valid)

    # Save debug image with green contours drawn — inspect at /static/processed.png
    debug = img.copy()
    cv2.drawContours(debug, valid, -1, (0, 255, 0), 2)
    os.makedirs("static", exist_ok=True)
    cv2.imwrite("static/processed.png", debug)

    return {
        "total_pixel_area": total_px,
        "surface_count": len(valid),
    }


# ---------------------------------------------------------------------------
# 4. Energy / finance estimator
# ---------------------------------------------------------------------------


def estimate_solar(area_sqm: float, lat: float, lon: float) -> dict:
    """
    Converts real-world usable area into solar energy estimates.
    Constants sourced from real engineering references for India.
    """
    USABILITY_FACTOR = 0.70  # not every m² is installable
    PANEL_AREA_SQM = 2.0  # standard 400W panel footprint
    PANEL_WATT_PEAK = 400  # watts per panel
    PEAK_SUN_HOURS = get_peak_sun_hours(
        lat, lon
    )  # India average (NASA POWER API gives exact value per lat/lon — future improvement)
    SYSTEM_EFFICIENCY = 0.80  # inverter + wiring + dust losses
    COST_PER_KW_INR = 65_000  # full installation with MNRE subsidy (2024)
    ELEC_RATE_INR = 8  # Rs/kWh average Indian household tariff
    CO2_FACTOR = 0.82  # kg CO2 per kWh (India grid emission factor)

    installable_area = area_sqm * USABILITY_FACTOR
    num_panels = installable_area / PANEL_AREA_SQM
    system_kw = (num_panels * PANEL_WATT_PEAK) / 1000
    kwh_day = system_kw * PEAK_SUN_HOURS * SYSTEM_EFFICIENCY
    kwh_year = kwh_day * 365
    total_cost = system_kw * COST_PER_KW_INR
    annual_savings = kwh_year * ELEC_RATE_INR
    payback_years = (total_cost / annual_savings) if annual_savings > 0 else 0
    roi_pct = (annual_savings / total_cost * 100) if total_cost > 0 else 0

    return {
        "equipment": "solar",
        "installable_area_sqm": round(installable_area, 1),
        "num_panels": round(num_panels),
        "system_size_kw": round(system_kw, 2),
        "energy_kwh_per_day": round(kwh_day, 1),
        "energy_kwh_per_year": round(kwh_year, 0),
        "total_cost_inr": round(total_cost, 0),
        "annual_savings_inr": round(annual_savings, 0),
        "payback_years": round(payback_years, 1),
        "roi_percent": round(roi_pct, 1),
        "co2_saved_kg_year": round(kwh_year * CO2_FACTOR, 0),
    }


# ---------------------------------------------------------------------------
# 5. Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json()

    lat = body.get("lat")
    lon = body.get("lon")
    zoom = body.get("zoom", 18)

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    try:
        # Step 1 — fetch satellite image (free, no API key)
        img = fetch_satellite_image(lat, lon, zoom=zoom)

        # Step 2 — real-world pixel scale
        mpp = meters_per_pixel(lat, zoom)

        # Step 3 — detect usable surfaces
        detection = detect_usable_surfaces(img)
        real_area_sqm = detection["total_pixel_area"] * (mpp**2)

        # Step 4 — energy estimates
        estimates = estimate_solar(real_area_sqm, lat, lon)

        return jsonify(
            {
                "status": "success",
                "location": {"lat": lat, "lon": lon},
                "detection": {
                    "area_sqm": round(real_area_sqm, 1),
                    "surface_count": detection["surface_count"],
                    "meters_per_pixel": round(mpp, 4),
                },
                "estimates": estimates,
            }
        )

    except requests.HTTPError as e:
        return jsonify({"error": f"Tile fetch error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    message = body.get("message", "")
    context = body.get("context", {})

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are Greenifier's AI assistant. A location has been analysed for solar energy potential.

Results:
- Installable area: {context.get("installable_area_sqm")} m²
- Solar panels: {context.get("num_panels")}
- System size: {context.get("system_size_kw")} kW
- Energy per day: {context.get("energy_kwh_per_day")} kWh
- Energy per year: {context.get("energy_kwh_per_year")} kWh
- Total cost: Rs.{context.get("total_cost_inr")}
- Annual savings: Rs.{context.get("annual_savings_inr")}
- ROI: {context.get("roi_percent")}% per year
- Payback period: {context.get("payback_years")} years
- CO2 saved per year: {context.get("co2_saved_kg_year")} kg

Answer simply and concisely using the numbers above.""",
            },
            {"role": "user", "content": message},
        ],
    )

    return jsonify({"reply": response.choices[0].message.content})


if __name__ == "__main__":
    app.run(debug=True)