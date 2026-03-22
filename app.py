import math
import os
import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# 1. Satellite image fetcher — free, no API key
# ---------------------------------------------------------------------------


def fetch_satellite_image(lat: float, lon: float, zoom: int = 18) -> np.ndarray:
    lat_r = math.radians(lat)
    n = 2**zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int(
        (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    )

    headers = {"User-Agent": "Mozilla/5.0"}

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

    stitched = cv2.vconcat([cv2.hconcat(row) for row in tiles])
    h, w = stitched.shape[:2]
    cx, cy = w // 2, h // 2
    img = stitched[cy - 320 : cy + 320, cx - 320 : cx + 320]
    return img


# ---------------------------------------------------------------------------
# 2. Ground resolution — pixels → real-world meters
# ---------------------------------------------------------------------------


def meters_per_pixel(lat: float, zoom: int = 18) -> float:
    earth_circumference = 40_075_016.686
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
        return 5.2


# ---------------------------------------------------------------------------
# 3. Roof / usable surface detector
# ---------------------------------------------------------------------------


def detect_usable_surfaces(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roof_mask = cv2.inRange(hsv, (0, 0, 120), (180, 55, 255))
    land_mask = cv2.inRange(hsv, (8, 20, 80), (32, 130, 210))
    combined = cv2.bitwise_or(roof_mask, land_mask)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick_edges = cv2.dilate(edges, kernel, iterations=2)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(thick_edges))

    clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, clean)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA_PX = 150
    MAX_AREA_PX = 70_000

    valid = [c for c in contours if MIN_AREA_PX < cv2.contourArea(c) < MAX_AREA_PX]
    total_px = sum(cv2.contourArea(c) for c in valid)

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
    USABILITY_FACTOR = 0.70
    PANEL_AREA_SQM = 2.0
    PANEL_WATT_PEAK = 400
    PEAK_SUN_HOURS = get_peak_sun_hours(lat, lon)
    SYSTEM_EFFICIENCY = 0.80
    COST_PER_KW_INR = 65_000
    ELEC_RATE_INR = 8
    CO2_FACTOR = 0.82

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


# ----------------------------------------------------
# GET WIND Speed
# ----------------------------------------------------------------------------
def get_wind_speed(lat: float, lon: float) -> float:
    """
    Fetches average wind speed at 10m height from Open-Meteo API.
    Free, no API key needed.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=windspeed_10m"
        "&wind_speed_unit=ms"
        "&forecast_days=1"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        speeds = data["hourly"]["windspeed_10m"]
        return round(sum(speeds) / len(speeds), 2)
    except Exception:
        return 3.0  # fallback average wind speed m/s


def estimate_wind(area_sqm: float, lat: float, lon: float) -> dict:
    """
    Estimates wind energy potential for a given area.
    Uses small wind turbines suitable for rooftops and open land.
    """
    wind_speed = get_wind_speed(lat, lon)
    print(f"WIND SPEED: {wind_speed}")
    if wind_speed < 3.5:
        return {
            "equipment": "wind",
            "wind_speed_ms": wind_speed,
            "viable": False,
            "message": f"Wind speed {wind_speed} m/s is too low for viable generation. Minimum recommended is 3.5 m/s.",
            "num_turbines": 0,
            "system_size_kw": 0,
            "energy_kwh_per_day": 0,
            "energy_kwh_per_year": 0,
            "total_cost_inr": 0,
            "annual_savings_inr": 0,
            "payback_years": 0,
            "roi_percent": 0,
            "co2_saved_kg_year": 0,
        }

    # Small wind turbine constants
    TURBINE_RATED_KW = 5.0  # kW per turbine (small rooftop/commercial)
    TURBINE_AREA_SQM = 200  # m² needed per turbine (clearance included)
    CAPACITY_FACTOR = 0.25  # realistic for urban/semi-urban India
    COST_PER_KW_INR = 80_000  # small wind turbine installation cost
    ELEC_RATE_INR = 8  # Rs/kWh
    CO2_FACTOR = 0.82  # kg CO2 per kWh

    # Wind power scales with cube of wind speed
    # Normalised against rated speed of 12 m/s
    RATED_WIND_SPEED = 12.0
    speed_factor = (wind_speed / RATED_WIND_SPEED) ** 3
    speed_factor = min(speed_factor, 1.0)  # cap at rated power

    num_turbines = int(area_sqm / TURBINE_AREA_SQM)
    system_kw = num_turbines * TURBINE_RATED_KW
    kwh_day = system_kw * 24 * CAPACITY_FACTOR * speed_factor
    kwh_year = kwh_day * 365
    total_cost = system_kw * COST_PER_KW_INR
    annual_savings = kwh_year * ELEC_RATE_INR
    payback_years = (total_cost / annual_savings) if annual_savings > 0 else 0
    roi_pct = (annual_savings / total_cost * 100) if total_cost > 0 else 0

    return {
        "equipment": "wind",
        "wind_speed_ms": wind_speed,
        "num_turbines": num_turbines,
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
# Equipment catalogue
# ---------------------------------------------------------------------------

EQUIPMENT_CATALOGUE = {
    "solar": {
        "longi_400w": {
            "name": "LONGi LR4-60HPH 400W",
            "watt_peak": 400,
            "area_sqm": 2.0,
            "cost_per_panel_inr": 18000,
            "efficiency": 0.21,
            "best_for": "rooftop",
        },
        "trina_550w": {
            "name": "Trina Vertex 550W",
            "watt_peak": 550,
            "area_sqm": 2.5,
            "cost_per_panel_inr": 24000,
            "efficiency": 0.21,
            "best_for": "rooftop",
        },
        "waaree_335w": {
            "name": "Waaree WS-335",
            "watt_peak": 335,
            "area_sqm": 1.9,
            "cost_per_panel_inr": 14000,
            "efficiency": 0.18,
            "best_for": "rooftop",
        },
        "first_solar_fs": {
            "name": "First Solar FS-6 110W",
            "watt_peak": 110,
            "area_sqm": 1.2,
            "cost_per_panel_inr": 8000,
            "efficiency": 0.11,
            "best_for": "hot_climate",
        },
        "sunpower_400w": {
            "name": "SunPower Maxeon 400W",
            "watt_peak": 400,
            "area_sqm": 1.8,
            "cost_per_panel_inr": 32000,
            "efficiency": 0.22,
            "best_for": "limited_space",
        },
    },
    "wind": {
        "suzlon_1kw": {
            "name": "Suzlon S111 1kW",
            "rated_kw": 1,
            "area_sqm": 50,
            "cost_inr": 85000,
            "min_wind_ms": 3.5,
            "best_for": "rooftop",
        },
        "wind_world_2kw": {
            "name": "Wind World W-2kW",
            "rated_kw": 2,
            "area_sqm": 100,
            "cost_inr": 160000,
            "min_wind_ms": 4.0,
            "best_for": "rooftop",
        },
        "vestas_5kw": {
            "name": "Vestas V15 5kW",
            "rated_kw": 5,
            "area_sqm": 200,
            "cost_inr": 350000,
            "min_wind_ms": 4.5,
            "best_for": "open_land",
        },
        "enercon_50kw": {
            "name": "Enercon E-33 50kW",
            "rated_kw": 50,
            "area_sqm": 500,
            "cost_inr": 3500000,
            "min_wind_ms": 5.0,
            "best_for": "open_land",
        },
        "ge_100kw": {
            "name": "GE 1.5MW (100kW unit)",
            "rated_kw": 100,
            "area_sqm": 1000,
            "cost_inr": 7000000,
            "min_wind_ms": 6.0,
            "best_for": "large_open_land",
        },
    },
    "rainwater": {
        "basic_rooftop": {
            "name": "Basic Rooftop Harvesting",
            "collection_sqm": 1,
            "cost_per_sqm_inr": 500,
            "storage_litres": 5000,
            "best_for": "rooftop",
        },
        "large_tank": {
            "name": "Underground Storage Tank",
            "collection_sqm": 1,
            "cost_per_sqm_inr": 1200,
            "storage_litres": 20000,
            "best_for": "open_land",
        },
    },
}


def recommend_equipment(
    area_sqm: float, wind_speed: float, lat: float, lon: float
) -> dict:
    """
    Agent 3 — recommends the best equipment based on location data.
    Logic:
    - Always recommend solar (works everywhere)
    - Recommend wind only if wind speed is viable
    - Recommend rainwater harvesting based on rainfall data
    - Pick best product within each category based on area and location
    """
    recommendations = {}

    # --- Solar recommendation ---
    # Pick panel based on area available
    if area_sqm < 100:
        panel_id = "sunpower_400w"  # limited space → highest efficiency
    elif lat > 20:
        panel_id = "waaree_335w"  # north India → affordable local brand
    else:
        panel_id = "longi_400w"  # default → best value globally

    recommendations["solar"] = {
        "recommended": True,
        "product_id": panel_id,
        "product": EQUIPMENT_CATALOGUE["solar"][panel_id]["name"],
        "reason": "Solar is viable at this location based on detected rooftop area and sunlight data.",
    }

    # --- Wind recommendation ---
    if wind_speed >= 6.0:
        turbine_id = "ge_100kw"
        reason = (
            f"Excellent wind speed ({wind_speed} m/s) — large turbines recommended."
        )
    elif wind_speed >= 5.0:
        turbine_id = "enercon_50kw"
        reason = f"Good wind speed ({wind_speed} m/s) — medium turbines viable."
    elif wind_speed >= 4.5:
        turbine_id = "vestas_5kw"
        reason = f"Moderate wind speed ({wind_speed} m/s) — small turbines recommended."
    elif wind_speed >= 4.0:
        turbine_id = "wind_world_2kw"
        reason = f"Low-moderate wind speed ({wind_speed} m/s) — micro turbines only."
    elif wind_speed >= 3.5:
        turbine_id = "suzlon_1kw"
        reason = (
            f"Marginal wind speed ({wind_speed} m/s) — only smallest turbines viable."
        )
    else:
        turbine_id = None
        reason = f"Wind speed {wind_speed} m/s is too low for any turbine."

    recommendations["wind"] = {
        "recommended": turbine_id is not None,
        "product_id": turbine_id,
        "product": EQUIPMENT_CATALOGUE["wind"][turbine_id]["name"]
        if turbine_id
        else None,
        "reason": reason,
    }

    # --- Rainwater recommendation ---
    # Simple rule: always recommend for rooftops, bigger system for larger areas
    if area_sqm >= 200:
        rw_id = "large_tank"
    else:
        rw_id = "basic_rooftop"

    recommendations["rainwater"] = {
        "recommended": True,
        "product_id": rw_id,
        "product": EQUIPMENT_CATALOGUE["rainwater"][rw_id]["name"],
        "reason": "Rainwater harvesting is viable at any location with a detected rooftop.",
    }

    return recommendations


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
        img = fetch_satellite_image(lat, lon, zoom=zoom)
        mpp = meters_per_pixel(lat, zoom)
        detection = detect_usable_surfaces(img)
        real_area_sqm = detection["total_pixel_area"] * (mpp**2)
        wind_data = estimate_wind(real_area_sqm, lat, lon)
        wind_speed = wind_data.get("wind_speed_ms", 0)

        estimates = {
            "solar": estimate_solar(real_area_sqm, lat, lon),
            "wind": wind_data,
        }

        recommendations = recommend_equipment(real_area_sqm, wind_speed, lat, lon)

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
                "recommendations": recommendations,
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

    system_prompt = f"""You are Greenifier's AI assistant. A location has been analysed for solar energy potential.

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

Answer questions simply and concisely using the numbers above."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        max_tokens=300,
    )

    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
