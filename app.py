import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Premier League Transfer Predictor", layout="centered"
)
st.title("⚽ Premier League Player Valuation Tool")
st.write(
    "Select any player to compare our machine learning valuation against their actual market value."
)

# 1. Load Data & Train Model
df = pd.read_csv("players.csv")

features = [
    "Age",
    "Goals",
    "Assists",
    "Minutes",
    "Contract_Years",
    "Clean_Sheets",
    "Tackles_Int",
    "Pos_GK",
    "Pos_DEF",
    "Pos_MID",
    "Pos_FW",
]

X = df[features]
y = df["Value_M"]

model = LinearRegression()
model.fit(X, y)

# 2. Player Search Bar / Dropdown
player_names = sorted(df["Player"].unique())
selected_player = st.selectbox(
    "Type or select a player name:",
    options=player_names,
    index=None,
    placeholder="Start typing a player name...",
)

# 3. Valuation Output & Graph
if selected_player:
  player_row = df[df["Player"] == selected_player].iloc[0]

  # Extract player stats and calculate model prediction
  player_stats = [player_row[f] for f in features]
  predicted_val = model.predict([player_stats])[0]
  actual_val = player_row["Value_M"]

  st.markdown("---")
  st.subheader(f"Valuation Report: {selected_player}")

  # Display metrics side-by-side
  col1, col2, col3 = st.columns(3)
  col1.metric("Model Value", f"£{predicted_val:.1f}M")
  col2.metric("Market Price", f"£{actual_val:.1f}M")

  diff = predicted_val - actual_val
  if diff > 0:
    col3.metric("Market Status", "Undervalued", delta=f"+£{diff:.1f}M")
  else:
    col3.metric("Market Status", "Overvalued", delta=f"-£{abs(diff):.1f}M")

  # Comparison Bar Chart
  st.write("### Value Comparison Chart")
  chart_data = pd.DataFrame(
      {
          "Valuation Type": ["Model Prediction", "Actual Market Price"],
          "Value (£M)": [round(predicted_val, 1), actual_val],
      }
  )

  st.bar_chart(data=chart_data, x="Valuation Type", y="Value (£M)")
