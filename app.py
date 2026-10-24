import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Premier League Transfer Predictor", layout="centered"
)
st.title("⚽ Premier League Player Valuation Tool")
st.write(
    "Search for any Premier League player to calculate their fair transfer market value and project future asset value."
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

# 2. Player Search Dropdown
player_names = sorted(df["Player"].unique())
selected_player = st.selectbox(
    "Type or select a player's name:",
    options=player_names,
    index=None,
    placeholder="Start typing a player name...",
)

# 3. Valuation Output & Analytics
if selected_player:
  player_row = df[df["Player"] == selected_player].iloc[0]

  # Extract stats and predict current value
  player_stats = [player_row[f] for f in features]
  predicted_val = model.predict([player_stats])[0]
  actual_val = player_row["Value_M"]

  st.markdown("---")
  st.subheader(f"Valuation Report: {selected_player}")

  # Metric Cards
  col1, col2, col3 = st.columns(3)
  col1.metric("Model Estimated Value", f"£{predicted_val:.1f}M")
  col2.metric("Actual Market Value", f"£{actual_val:.1f}M")

  diff = predicted_val - actual_val
  if diff > 0:
    col3.metric("Market Status", "Undervalued", delta=f"+£{diff:.1f}M")
  else:
    col3.metric("Market Status", "Overvalued", delta=f"-£{abs(diff):.1f}M")

  # --- BAR CHART COMPARISON ---
  st.write("### Current Market Price vs. Model Estimate")
  bar_data = pd.DataFrame(
      {
          "Valuation Type": ["Model Prediction", "Actual Market Price"],
          "Value (£M)": [round(predicted_val, 1), actual_val],
      }
  )
  st.bar_chart(data=bar_data, x="Valuation Type", y="Value (£M)")

  # --- 5-YEAR FUTURE VALUE LINE GRAPH ---
  st.write("### 📉 5-Year Projected Market Value Curve")

  future_years = ["Current (2026)", "2027", "2028", "2029", "2030"]
  projected_values = []

  # Copy original player features for aging projection
  curr_age = player_row["Age"]
  curr_contract = player_row["Contract_Years"]

  for year_offset in range(5):
    # Model age aging +1 each year
    projected_age = curr_age + year_offset
    # Contract decreases by 1 each year (minimum 0)
    projected_contract = max(0, curr_contract - year_offset)

    # Build feature row with updated Age and Contract_Years
    temp_stats = [
        projected_age,
        player_row["Goals"],
        player_row["Assists"],
        player_row["Minutes"],
        projected_contract,
        player_row["Clean_Sheets"],
        player_row["Tackles_Int"],
        player_row["Pos_GK"],
        player_row["Pos_DEF"],
        player_row["Pos_MID"],
        player_row["Pos_FW"],
    ]

    # Predict future value (floor at 0.0M)
    val = max(0.0, model.predict([temp_stats])[0])
    projected_values.append(round(val, 1))

  # Build DataFrame for line chart
  line_data = pd.DataFrame(
      {"Year": future_years, "Projected Value (£M)": projected_values}
  )

  # Display Line Chart
  st.line_chart(data=line_data, x="Year", y="Projected Value (£M)")
