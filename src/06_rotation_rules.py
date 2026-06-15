"""
06_rotation_rules.py  (Version 3)

Responsible for:
1. STATE–CROP SUITABILITY RULES
   - Prevents nonsense recommendations like Orange in Rajasthan.
2. SIMPLE CLIMATE FILTER
   - Uses rainfall & temperature thresholds per crop.
3. BASIC CROP ROTATION SUGGESTIONS
   - Suggest next-season crops based on current crop family.

This module does NOT train any ML model.
It only receives crops predicted by the classifier and filters / adjusts them.
"""

from typing import List

# ----------------------------------------------------
# 1. State-wise allowed crops  (you can extend anytime)
# ----------------------------------------------------
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
    ]
}

# ----------------------------------------------------
# 2. Simple climate thresholds per crop
#    (very coarse; improves realism)
# ----------------------------------------------------
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
    s = str(state).strip().title()
    return s if s in STATE_ALLOWED_CROPS else "OTHER"


def filter_by_state(predicted_crops: List[str], state: str) -> List[str]:
    """Keep only those crops which are allowed in that state."""
    norm_state = _normalize_state_name(state)
    allowed = STATE_ALLOWED_CROPS.get(norm_state, STATE_ALLOWED_CROPS["OTHER"])

    filtered = [c for c in predicted_crops if c in allowed]

    # If we removed too many, back-fill with allowed crops
    if len(filtered) < len(predicted_crops):
        for c in allowed:
            if c not in filtered:
                filtered.append(c)
            if len(filtered) == len(predicted_crops):
                break
    return filtered


def climate_ok(crop: str, rainfall_mm: float, temperature_c: float) -> bool:
    """Check if rainfall & temperature are roughly in range for this crop."""
    rule = CLIMATE_RULES.get(crop)
    if not rule:
        return True  # no rule → accept

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
    Main function to call from app.py:

    1. Take Top-N crops from ML model.
    2. Filter by state.
    3. Filter by climate.
    4. Return final Top-K crops.
    """
    if not predicted_crops:
        return []

    # 1. State filter
    filtered = filter_by_state(predicted_crops, state)

    # 2. Climate filter
    climate_filtered = [c for c in filtered if climate_ok(c, rainfall_mm, temperature_c)]

    # If all removed by climate filter, fall back to state-only
    if not climate_filtered:
        climate_filtered = filtered

    return climate_filtered[:top_k]


# ----------------------------------------------------
# 3. Crop Rotation Rules (simple version)
# ----------------------------------------------------
ROTATION_RULES = {
    "Rice":      ["Mustard", "Potato", "Pulses"],
    "Wheat":     ["Gram", "Mustard", "Sunflower", "Pulses"],
    "Maize":     ["Legumes", "Groundnut", "Green Gram"],
    "Cotton":    ["Pulses", "Wheat", "Oilseeds"],
    "Sugarcane": ["Pulses", "Vegetables"],
    "Pulses":    ["Cereals (Rice/Wheat/Maize)"],
    "Jute":      ["Paddy", "Pulses"],
}


def suggest_rotation(current_crop: str):
    """Return a list of recommended next crops for rotation."""
    key = str(current_crop).strip().title()
    return ROTATION_RULES.get(key, ["Rotate with a legume crop to improve soil nitrogen."])
