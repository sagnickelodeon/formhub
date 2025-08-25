# Formula 1 Teammate-Based Analytics Hub
_Data-driven driver insights where every comparison is car-neutral_

---

## Table of Contents
1. [About](#about)  
2. [Features](#features)  
3. [Quick Start](#quick-start)  
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)  
6. [License](#license)

---

## About
This repository contains a **Streamlit** application that converts raw FIA timing data into interactive dashboards, rankings and comparisons.  
All statistics are ***teammate-normalised*** — drivers are compared **only against the person in the other seat of the same car**, eliminating chassis and engine bias.

Typical questions the app can answer:

* *“Which driver makes the biggest grid-to-flag comebacks?”*  
* *“Was 2021 or 2024 the more dramatic season?”*  
* *“Which constructor has improved reliability the most in the hybrid era?”*

---

## Features

| Module | What you get | Example questions |
|--------|--------------|-------------------|
| **⚔️ Head-to-Head** | Choose any two drivers and see cumulative wins, battle log, victory margins, grid deficits and Elo swings. | *Did Rosberg ever lead Hamilton mid-season?* |
| **📈 Performance Analytics** | 4 tabs (Comeback, Race Craft, Form Guide, Consistency) with safe filters and Plotly charts. | *Who gains the most places on Sundays?* |
| **🏁 Race Statistics** | Grand-Prix records, season dashboards, era trends, margin vs rating scatter. | *Which track has the highest comeback rate?* |
| **🔧 Reliability** | Driver DNF timing, lap-completion consistency, constructor DNF trend lines. | *Is Ferrari more reliable now than five years ago?* |
| **🧮 Live Elo Ratings** | Bayesian Elo updated after every teammate duel with shrinking K-factor. | *How steep was Piastri’s rookie curve?* |
| **🛡 Safe-Filter Engine** | Year/team sliders fall back to sane defaults so you never see an empty screen. |
| **⚡ Responsive UI** | Hover tool-tips, legend click-to-hide, dark-mode friendly. |
| **📦 Modular Code** | Each page lives in `pages/`; add a new tab in <50 LOC. |

---

## Quick Start

git clone https://github.com/<your-user>/f1-teammate-analytics.git
cd f1-teammate-analytics

### optional: create virtual environment
`python -m venv .venv`
`source .venv/bin/activate # Windows: .venv\Scripts\activate`

`pip install -r requirements.txt`

`streamlit run main.py # open http://localhost:8501`


First launch downloads public race CSVs (~2 MB) and builds the initial rating table.

---

## Project Structure

<pre>
├── main.py # Streamlit entry point
├── streamlit_pages/
│ ├── about.py
│ ├── advanced_stats.py (For certain reasons this page is only visible to the admin in the UI)
│ ├── driver_ranking.py
│ ├── head_to_head.py
│ ├── performance_analytics.py
│ ├── race_statistics.py
│ ├── reliability_analysis.py
│ └── track_analytics.py
├── util.py
├── requirements.txt
└── README.md
</pre>
  
## Contributing
1. Fork ➜ `git checkout -b feature/my-improvement`  
2. Commit changes with clear messages  
3. Ensure `pytest` & `ruff` pass  
4. Open a Pull Request 🎉

---

## License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE).

> **Data remains property of its original providers.**  
> Use this repository for educational and personal analysis only.

---

**Enjoy exploring — and remember:** in F1 the first rival you must beat is the one in the *same* car.
