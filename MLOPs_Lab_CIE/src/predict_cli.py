import argparse
import joblib
import pandas as pd

# Load model (we will save it later if needed)
MODEL_PATH = "models/model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(args):
    model = load_model()

    data = pd.DataFrame([{
        "panel_area_sqm": args.panel_area_sqm,
        "sunlight_hours": args.sunlight_hours,
        "cloud_cover_pct": args.cloud_cover_pct,
        "panel_age_years": args.panel_age_years
    }])

    prediction = model.predict(data)[0]
    print(round(prediction, 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--panel_area_sqm", type=float, required=True)
    parser.add_argument("--sunlight_hours", type=float, required=True)
    parser.add_argument("--cloud_cover_pct", type=float, required=True)
    parser.add_argument("--panel_age_years", type=float, required=True)

    args = parser.parse_args()
    predict(args)