"""Streamlit user interface for the REP incident-grade classifier.

Run from the repository root with:
	streamlit run ui/app.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_store"
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
ENCODER_PATH = MODEL_DIR / "feature_encoder.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"


st.set_page_config(
	page_title="REP | Incident Grade Predictor",
	page_icon="🛡️",
	layout="wide",
)


@st.cache_resource
def load_artifacts():
	missing = [
		path.name
		for path in (MODEL_PATH, FEATURES_PATH, ENCODER_PATH, TARGET_ENCODER_PATH)
		if not path.exists()
	]
	if missing:
		raise FileNotFoundError(
			"Missing model artifacts: " + ", ".join(missing) + 
			". Run training/train_model.py first."
		)

	with FEATURES_PATH.open(encoding="utf-8") as file:
		feature_columns = json.load(file)

	return (
		joblib.load(MODEL_PATH),
		joblib.load(ENCODER_PATH),
		joblib.load(TARGET_ENCODER_PATH),
		feature_columns,
	)


try:
	model, feature_encoder, target_encoder, feature_columns = load_artifacts()
except Exception as error:
	st.error(str(error))
	st.stop()


CATEGORICAL_FEATURES = set(feature_encoder.feature_names_in_)
NUMERIC_FEATURES = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]


def numeric_input(label: str, default: float = 0) -> float:
	return st.number_input(label, value=float(default), step=1.0, format="%.0f")


def build_input() -> pd.DataFrame:
	"""Collect one incident and return it in the training feature order."""
	values = {}

	with st.form("incident_form"):
		st.subheader("Incident details")
		left, middle, right = st.columns(3)

		with left:
			values["DetectorId"] = numeric_input("Detector ID")
			values["AlertTitle"] = numeric_input("Alert title ID")
			values["ApplicationName"] = numeric_input("Application name ID")
			values["FileName"] = numeric_input("File name ID")
			values["FolderPath"] = numeric_input("Folder path ID")
			values["CountryCode"] = numeric_input("Country code")
			values["State"] = numeric_input("State")
			values["City"] = numeric_input("City")

		with middle:
			for column in ("Category", "EntityType", "EvidenceRole", "ResourceType"):
				values[column] = st.text_input(column, value="Unknown")
			values["OSFamily"] = numeric_input("OS family")
			values["OSVersion"] = numeric_input("OS version")
			values["Hour"] = st.slider("Hour", 0, 23, 12)
			values["DayOfWeek"] = st.slider("Day of week", 0, 6, 2)

		with right:
			for column in (
				"MitreTechniques", "ActionGrouped", "ActionGranular",
				"ThreatFamily", "Roles", "AntispamDirection",
				"SuspicionLevel", "LastVerdict",
			):
				values[column] = st.text_input(column, value="Unknown")
			values["Month"] = st.slider("Month", 1, 12, 6)
			values["IsWeekend"] = st.checkbox("Weekend", value=False)

		submitted = st.form_submit_button("Predict incident grade", type="primary")

	if not submitted:
		return pd.DataFrame()

	row = pd.DataFrame([values]).reindex(columns=feature_columns)
	row[NUMERIC_FEATURES] = row[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
	if CATEGORICAL_FEATURES:
		row[list(CATEGORICAL_FEATURES)] = feature_encoder.transform(row[list(CATEGORICAL_FEATURES)])
	return row.fillna(0)


st.title("🛡️ REP Incident Grade Predictor")
st.caption("Random Forest classification service for security incident triage")

input_data = build_input()
if not input_data.empty:
	predicted_index = int(model.predict(input_data)[0])
	probabilities = model.predict_proba(input_data)[0]
	labels = target_encoder.inverse_transform(model.classes_)
	predicted_label = target_encoder.inverse_transform([predicted_index])[0]

	st.divider()
	st.subheader("Prediction")
	st.success(f"Predicted incident grade: **{predicted_label}**")

	probability_df = pd.DataFrame({
		"Incident grade": labels,
		"Probability": probabilities,
	}).sort_values("Probability", ascending=False)
	probability_df["Probability"] = probability_df["Probability"].map(lambda value: f"{value:.2%}")
	st.dataframe(probability_df, hide_index=True, use_container_width=True)

	st.download_button(
		"Download prediction JSON",
		data=pd.Series({
			"predicted_grade": predicted_label,
			"probabilities": dict(zip(labels, np.round(probabilities, 6))),
		}).to_json(indent=2),
		file_name="rep_prediction.json",
		mime="application/json",
	)
