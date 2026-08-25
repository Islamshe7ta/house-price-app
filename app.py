import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import altair as alt

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown("""
<style>
    /* Clean professional header without aggressive background colors that break contrast */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1e3c72;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-header {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AI House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional real estate valuation based on Machine Learning models.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_resources_v4():
    try:
        model = joblib.load("models/house_price.pkl")
        with open("models/locations.json", "r") as f:
            locations = json.load(f)
            
        df = pd.read_csv("data/house_prices.csv")
        
        def parse_amount(x):
            if not isinstance(x, str):
                return None
            x = x.strip().lower()
            try:
                if "lac" in x:
                    return float(x.replace("lac", "").strip()) * 1e5
                if "cr" in x:
                    return float(x.replace("cr", "").strip()) * 1e7
                return float(x.replace(",", ""))
            except ValueError:
                return None
                
        df["price"] = df["Amount(in rupees)"].apply(parse_amount)
        return model, locations, df, None
    except Exception as e:
        import traceback
        return None, [], None, traceback.format_exc()

model, locations, df_raw, error_msg = load_resources_v4()

if not model:
    st.error("Model files missing or error loading resources. Please check the logs.")
    if error_msg:
        st.code(error_msg, language="python")
else:
    with st.form("prediction_form", border=True):
        st.subheader("Property Specifications")
        
        # Horizontal layout with 3 columns
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            area = st.number_input("Carpet Area (sqft)", min_value=100.0, max_value=20000.0, value=1200.0, step=100.0)
            floor = st.number_input("Floor Number (0=Ground, -1=Basement)", min_value=-5, max_value=100, value=1)
            bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
            
        with col2:
            balcony = st.number_input("Number of Balconies", min_value=0, max_value=10, value=1)
            location = st.selectbox("Location Area", options=locations, index=locations.index("Other") if "Other" in locations else 0)
            furnishing = st.selectbox("Furnishing Status", options=["Unfurnished", "Semi-Furnished", "Furnished"])
            
        with col3:
            transaction = st.selectbox("Transaction Type", options=["New Property", "Resale"])
            facing = st.selectbox("Facing Direction", options=["East", "West", "North", "South", "North-East", "North-West", "South-East", "South-West"])
            ownership = st.selectbox("Ownership Type", options=["Freehold", "Leasehold", "Co-operative Society", "Power of Attorney"])
            
        submit_button = st.form_submit_button(label="Calculate Estimated Price")
        
    if submit_button:
        input_data = {
            "carpet_area_sqft": area,
            "floor_num": floor,
            "Bathroom": bathrooms,
            "Balcony": balcony,
            "location_grouped": location,
            "Furnishing": furnishing,
            "Transaction": transaction,
            "Ownership": ownership,
            "facing": facing
        }
        
        input_df = pd.DataFrame([input_data])
        
        with st.spinner("Analyzing market data..."):
            try:
                pred_log = model.predict(input_df)
                pred_price = np.expm1(pred_log)[0]
                
                st.divider()
                st.subheader("Valuation Result")
                
                res_col1, res_col2 = st.columns(2)
                
                display_cr = f"{pred_price / 10000000:,.2f} Cr" if pred_price > 10000000 else "N/A"
                display_lac = f"{pred_price / 100000:,.2f} Lacs"
                
                with res_col1:
                    st.metric("Estimated Value (Lacs)", display_lac)
                with res_col2:
                    st.metric("Estimated Value (Crores)", display_cr if display_cr != "N/A" else "—")
                
                st.markdown(f"**Exact Prediction Calculation:** INR {pred_price:,.0f}")
                
                # Visualizations
                st.divider()
                st.subheader("Market Comparison")
                
                chart_col1, chart_col2 = st.columns(2)
                
                # Filter dataset for the selected location to compare
                if location != "Other" and df_raw is not None:
                    loc_df = df_raw[df_raw["location"] == location].dropna(subset=["price"])
                    
                    if len(loc_df) > 5:
                        with chart_col1:
                            st.markdown(f"**Price Distribution in {location}**")
                            # Create a histogram using Altair
                            hist = alt.Chart(loc_df).mark_bar(opacity=0.7, color="#2a5298").encode(
                                alt.X("price:Q", bin=alt.Bin(maxbins=20), title="Price (INR)"),
                                alt.Y('count()', title="Number of Properties")
                            )
                            # Add a vertical rule for the predicted price
                            rule = alt.Chart(pd.DataFrame({'price': [pred_price]})).mark_rule(color='red', size=2).encode(x='price:Q')
                            
                            st.altair_chart((hist + rule).interactive(), use_container_width=True)
                            st.caption("Red line indicates your estimated property value.")
                            
                # Area comparison
                if df_raw is not None:
                    with chart_col2:
                        st.markdown("**Price vs Carpet Area (Market Overview)**")
                        # We will take a sample to avoid heavy rendering
                        sample_df = df_raw.dropna(subset=["price", "Carpet Area"]).sample(n=min(1000, len(df_raw)))
                        # Ensure carpet area is numeric for the sample
                        sample_df["area_num"] = sample_df["Carpet Area"].apply(
                            lambda x: float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if any(c.isdigit() for c in str(x)) else None
                        )
                        sample_df = sample_df.dropna(subset=["area_num"])
                        
                        scatter = alt.Chart(sample_df).mark_circle(opacity=0.4, color="#1e3c72").encode(
                            x=alt.X("area_num:Q", scale=alt.Scale(type="log"), title="Carpet Area (sqft)"),
                            y=alt.Y("price:Q", scale=alt.Scale(type="log"), title="Price (INR)"),
                            tooltip=["price", "area_num"]
                        )
                        
                        # Add the predicted point
                        point = alt.Chart(pd.DataFrame({'area_num': [area], 'price': [pred_price]})).mark_point(
                            color='red', size=150, shape='cross', filled=True
                        ).encode(
                            x='area_num:Q',
                            y='price:Q'
                        )
                        
                        st.altair_chart((scatter + point).interactive(), use_container_width=True)
                        st.caption("Red cross indicates your estimated property value relative to the market.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

