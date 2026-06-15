"""
rules_state_climate.py  (Version 3 helper)

This module:
- Filters the ML-predicted Top-3 crops based on:
  1) State-wise allowed crops
  2) Rough rainfall & temperature suitability

Use:
    from rules_state_climate import apply_all_filters

    final_top3 = apply_all_filters(
        predicted_crops=raw_top3,
        state=state_name,
        rainfall_mm=rainfall,
        temperature_c=temp,
        top_k=3
    )
"""

from typing import List

# -----------------------------
# 1. State-wise allowed crops
# -----------------------------
STATE_ALLOWED_CROPS = {
    "Rajasthan": [
        "Wheat", "Bajra", "Gram", "Mustard", "Cotton",
        "Barley", "Guar", "Maize", "Pulses"
    ],
    "Punjab": [
        "Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Barley"
    ],
    "Haryana": [
        "Wheat", "Rice", "Sugarcane", "Cotton", "Mustard", "Pulses"
    ],
    "Uttar Pradesh": [
        "Wheat", "Rice", "Sugarcane", "Potato", "Pulses", "Maize", "Mustard"
    ],
    "West Bengal": [
        "Rice", "Potato", "Jute", "Mustard", "Sugarcane", "Maize", "Pulses"
    ],
    "Bihar": [
        "Rice", "Wheat", "Maize", "Pulses", "Sugarcane"
    ],
    "Madhya Pradesh": [
        "Soybean", "Wheat", "Gram", "Maize", "Pulses"
    ],
    "Maharashtra": [
        "Cotton", "Sugarcane", "Soybean", "Sorghum", "Maize", "Groundnut"
    ],
    "Gujarat": [
        "Groundnut", "Cotton", "Maize", "Wheat", "Bajra"
    ],
    "Karnataka": [
        "Millets", "Maize", "Cotton", "Sugarcane", "Paddy", "Groundnut"
    ],
    "Tamil Nadu": [
        "Rice", "Millets", "Sugarcane", "Cotton", "Groundnut", "Pulses"
    ],
    "Kerala": [
        "Coconut", "Rubber", "Tea", "Coffee", "Pepper"
    ],
    "Andhra Pradesh": [
        "Rice", "Groundnut", "Cotton", "Maize", "Sugarcane", "Pulses"
    ],
    "Assam": [
        "Rice", "Tea", "Jute", "Mustard", "Sugarcane"
    ],
    "OTHER": [
        "Rice", "Wheat", "Maize", "Pulses", "Millets"
    ],
}

# -----------------------------
# 2. Climate suitability rules
# -----------------------------
CLIMATE_RULES = {
    "Rice":      {"min_rain": 800, "max_rain": 3000, "min_temp": 18, "max_temp": 38},
    "Wheat":     {"min_rain": 300, "max_rain": 1200, "min_temp": 10, "max_temp": 25},
    "Maize":     {"min_rain": 500, "max_rain": 2000, "min_temp": 18, "max_temp": 35},
    "Bajra":     {"min_rain": 250, "max_rain": 800,  "min_temp": 20, "max_temp": 40},
    "Millets":   {"min_rain": 250, "max_rain": 900,  "min_temp": 20, "max_temp": 42},
    "Pulses":    {"min_rain": 300, "max_rain": 1200, "min_temp": 15, "max_temp": 35},
    "Cotton":    {"min_rain": 500, "max_rain": 1500, "min_temp": 20, "max_temp": 35},
    "Sugarcane": {"min_rain": 800, "max_rain": 2500, "min_temp": 20, "max_temp": 38},
    "Jute":      {"min_rain": 1200,"max_rain": 3000, "min_temp": 24, "max_temp": 35},
    "Tea":       {"min_rain": 1200,"max_rain": 3000, "min_temp": 14, "max_temp": 28},
    "Potato":    {"min_rain": 500, "max_rain": 1500, "min_temp": 15, "max_temp": 25},
    "Soybean":   {"min_rain": 600, "max_rain": 1500, "min_temp": 20, "max_temp": 30},
    "Mustard":   {"min_rain": 300, "max_rain": 800,  "min_temp": 10, "max_temp": 25},
}

def _normalize_state_name(state: str) -> str:
    if not state:
        return "OTHER"
    s = str(state).strip()
    # We assume your OHE categories already use nice capitalization, so:
    return s if s in STATE_ALLOWED_CROPS else "OTHER"


def filter_by_state(predicted_crops: List[str], state: str) -> List[str]:
    norm_state = _normalize_state_name(state)
    allowed = STATE_ALLOWED_CROPS.get(norm_state, STATE_ALLOWED_CROPS["OTHER"])

    filtered = [c for c in predicted_crops if c in allowed]

    # Backfill if we removed too many
    if len(filtered) < len(predicted_crops):
        for c in allowed:
            if c not in filtered:
                filtered.append(c)
            if len(filtered) == len(predicted_crops):
                break
    return filtered


def climate_ok(crop: str, rainfall_mm: float, temperature_c: float) -> bool:
    rule = CLIMATE_RULES.get(crop)
    if not rule:
        return True
    if rainfall_mm < rule["min_rain"] or rainfall_mm > rule["max_rain"]:
        return False
    if temperature_c < rule["min_temp"] or temperature_c > rule["max_temp"]:
        return False
    return True


def apply_all_filters(
    predicted_crops: List[str],
    state: str,
    rainfall_mm: float,
    temperature_c: float,
    top_k: int = 3,
) -> List[str]:
    """
    1. Filter by state
    2. Filter by climate
    3. Return up to top_k crops
    """
    if not predicted_crops:
        return []

    state_filtered = filter_by_state(predicted_crops, state)
    climate_filtered = [c for c in state_filtered if climate_ok(c, rainfall_mm, temperature_c)]

    if not climate_filtered:
        climate_filtered = state_filtered

    return climate_filtered[:top_k]
def get_allowed_crops(state: str):
    """Expose state-wise allowed crops for topping up Top-3 list."""
    return STATE_ALLOWED_CROPS.get(_normalize_state_name(state), STATE_ALLOWED_CROPS["OTHER"])


