# 🌿 Greenifier

**Greenifier** is an AI-powered web application that estimates renewable energy potential for any location on Earth using satellite imagery, computer vision, and large language models.

Click anywhere on the map → the system fetches real satellite imagery, detects usable rooftops and open surfaces, queries NASA for location-specific solar data, and returns a complete financial analysis — all in seconds.

---

## 🚀 Live Demo

> Coming soon — deployment in progress.

---

## ✨ Features

- **Satellite image analysis** — fetches real imagery at rooftop resolution, no API key required
- **AI roof detection** — OpenCV detects flat rooftops and open land using HSV colour thresholding and morphological processing
- **Real solar irradiance data** — queries NASA POWER API for exact peak sun hours at any lat/lon globally
- **Complete financial analysis** — system size, energy output, installation cost, ROI, payback period, CO₂ offset
- **AI chat assistant** — ask questions about the analysis in plain language, powered by Llama 3.3 via Groq
- **Debug view** — inspect detected surfaces at `/static/processed.png`

---

## 🧠 Architecture

```
User clicks map
      ↓
Frontend sends lat/lon → Flask backend
      ↓
Backend fetches 3×3 satellite tile grid (free, keyless)
      ↓
OpenCV detects usable surfaces → pixel area
      ↓
Web Mercator math → real square metres
      ↓
NASA POWER API → real solar irradiance for that location
      ↓
Energy + finance estimator → full analysis
      ↓
Groq LLM → answers user questions about the results
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Leaflet.js |
| Backend | Python, Flask, Flask-CORS |
| Computer Vision | OpenCV, NumPy |
| Solar Data | NASA POWER API |
| AI Chat | Groq API (Llama 3.3 70B) |
| Maps | Google Satellite Tiles (free, keyless) |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Lion4467cat/greenifier.git
cd greenifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the project root
```
GROQ_API_KEY=your-groq-api-key-here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the app
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 📊 How the Estimates Work

| Parameter | Value | Source |
|---|---|---|
| Panel size | 2 m² / 400W | Standard residential panel |
| Usability factor | 70% | Not every surface is installable |
| Peak sun hours | Variable | NASA POWER API per lat/lon |
| System efficiency | 80% | Inverter + wiring + dust losses |
| Installation cost | ₹65,000/kW | MNRE subsidy rates (India, 2024) |
| Electricity rate | ₹8/kWh | Average Indian household tariff |
| CO₂ factor | 0.82 kg/kWh | India grid emission factor |

---

## 🗺️ Roadmap

- [ ] Wind energy analysis
- [ ] Micro-hydro potential detection
- [ ] Agent 3 — equipment placement advisor (tilt angle, orientation)
- [ ] Full agentic orchestration with tool calling
- [ ] IBM Watson ML model for accurate roof segmentation
- [ ] Multi-location comparison table
- [ ] PDF report export
- [ ] Public deployment

---

## 👥 Team

- **S S Gokula Swamy** — Backend, computer vision, AI integration
- **Jiya Nitturkar** — Frontend, UI/UX design

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🌍 Why Greenifier?

Google Sunroof covers only solar panels in ~40 countries. Greenifier is designed to work globally — including data-sparse regions across South Asia, Southeast Asia, and Sub-Saharan Africa — and supports multiple green energy types. The countries that need renewable energy analysis the most are exactly the ones existing tools ignore.