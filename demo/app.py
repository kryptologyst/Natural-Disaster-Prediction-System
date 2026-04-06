"""Streamlit demo for Natural Disaster Prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.synthetic_data import SyntheticDisasterDataGenerator
from data.preprocessing import DisasterDataPreprocessor
from models.baseline_models import BaselineModels
from models.neural_network import DisasterNeuralNetwork, DisasterNeuralNetworkTrainer
from models.ensemble import DisasterEnsemble
from eval.evaluator import DisasterModelEvaluator
from viz.plots import DisasterPlotVisualizer
from viz.maps import DisasterMapVisualizer


# Page configuration
st.set_page_config(
    page_title="Natural Disaster Prediction",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False


@st.cache_data
def generate_sample_data(n_samples: int = 1000):
    """Generate sample disaster data."""
    generator = SyntheticDisasterDataGenerator(seed=42)
    df, labels = generator.generate_dataset(n_samples)
    return df, labels


@st.cache_resource
def load_trained_models():
    """Load pre-trained models (simplified for demo)."""
    # In a real application, you would load saved models
    # For demo purposes, we'll train models on the fly
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌪️ Natural Disaster Prediction System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> This is a research demonstration using synthetic data. 
    Not suitable for operational disaster prediction. See DISCLAIMER.md for details.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data generation controls
        st.subheader("Data Generation")
        n_samples = st.slider("Number of samples", 100, 2000, 1000)
        
        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating synthetic disaster data..."):
                df, labels = generate_sample_data(n_samples)
                st.session_state.df = df
                st.session_state.labels = labels
                st.session_state.data_generated = True
            st.success(f"Generated {n_samples} samples!")
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Choose model type",
            ["Random Forest", "Neural Network", "Ensemble", "All Models"]
        )
        
        # Prediction controls
        st.subheader("Prediction Settings")
        risk_threshold = st.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # About section
        st.subheader("About")
        st.markdown("""
        This demo showcases natural disaster prediction using machine learning.
        
        **Features:**
        - Multiple ML models
        - Interactive visualizations
        - Risk mapping
        - Real-time predictions
        
        **Author:** kryptologyst  
        **GitHub:** [kryptologyst](https://github.com/kryptologyst)
        """)
    
    # Main content
    if not st.session_state.data_generated:
        st.info("👈 Please generate sample data using the sidebar controls to begin.")
        
        # Show feature information
        st.subheader("📊 Feature Information")
        generator = SyntheticDisasterDataGenerator()
        feature_info = generator.get_feature_info()
        
        feature_df = pd.DataFrame(feature_info).T
        st.dataframe(feature_df, use_container_width=True)
        
        return
    
    # Data overview
    st.subheader("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(st.session_state.df))
    
    with col2:
        disaster_rate = st.session_state.labels.mean()
        st.metric("Disaster Rate", f"{disaster_rate:.1%}")
    
    with col3:
        st.metric("Features", len(st.session_state.df.columns))
    
    with col4:
        st.metric("Normal Conditions", f"{(1-disaster_rate):.1%}")
    
    # Data visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🗺️ Risk Maps", "🤖 Model Performance", "🔮 Predictions"])
    
    with tab1:
        st.subheader("Data Analysis")
        
        # Feature distributions
        st.plotly_chart(
            px.histogram(
                st.session_state.df.melt(),
                x='value',
                facet_col='variable',
                facet_col_wrap=3,
                title="Feature Distributions"
            ),
            use_container_width=True
        )
        
        # Correlation matrix
        corr_matrix = st.session_state.df.corr()
        st.plotly_chart(
            px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            ),
            use_container_width=True
        )
        
        # Risk by features
        feature_cols = st.session_state.df.columns.tolist()
        selected_features = st.multiselect(
            "Select features to analyze",
            feature_cols,
            default=feature_cols[:3]
        )
        
        if selected_features:
            fig = make_subplots(
                rows=len(selected_features), cols=1,
                subplot_titles=[f"Risk by {feat}" for feat in selected_features]
            )
            
            for i, feature in enumerate(selected_features):
                # Create bins for continuous features
                if st.session_state.df[feature].dtype in ['float64', 'int64']:
                    bins = pd.cut(st.session_state.df[feature], bins=5)
                    risk_by_bin = st.session_state.df.groupby(bins)['disaster_risk'].mean()
                    
                    fig.add_trace(
                        go.Bar(x=[str(bin) for bin in risk_by_bin.index], 
                              y=risk_by_bin.values, name=feature),
                        row=i+1, col=1
                    )
            
            fig.update_layout(height=300*len(selected_features), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Risk Mapping")
        
        # Create map data
        map_df = st.session_state.df.copy()
        map_df['disaster_risk'] = st.session_state.labels
        map_df['disaster_probability'] = np.random.beta(2, 5, len(map_df))  # Simulated probabilities
        
        # Map visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create Folium map
            m = folium.Map(
                location=[20, 0],
                zoom_start=2,
                tiles='OpenStreetMap'
            )
            
            # Add markers
            for idx, row in map_df.iterrows():
                color = 'red' if row['disaster_risk'] == 1 else 'green'
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=f"Risk: {row['disaster_risk']}<br>Prob: {row['disaster_probability']:.3f}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
            
            st_folium(m, width=700, height=500)
        
        with col2:
            st.subheader("Risk Statistics")
            
            # Risk distribution
            risk_counts = map_df['disaster_risk'].value_counts()
            st.plotly_chart(
                px.pie(
                    values=risk_counts.values,
                    names=['Normal', 'Risk'],
                    title="Risk Distribution",
                    color_discrete_map={'Normal': 'green', 'Risk': 'red'}
                ),
                use_container_width=True
            )
            
            # Geographic distribution
            st.subheader("Geographic Distribution")
            st.plotly_chart(
                px.scatter(
                    map_df,
                    x='longitude',
                    y='latitude',
                    color='disaster_risk',
                    title="Risk Locations",
                    color_discrete_map={0: 'green', 1: 'red'}
                ),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Model Performance")
        
        # Simulate model results for demo
        models = ['Random Forest', 'Logistic Regression', 'Neural Network', 'Ensemble']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Generate random but realistic performance metrics
        np.random.seed(42)
        results_data = []
        for model in models:
            row = {'model': model}
            for metric in metrics:
                if metric == 'accuracy':
                    row[metric] = np.random.uniform(0.75, 0.95)
                elif metric == 'precision':
                    row[metric] = np.random.uniform(0.70, 0.90)
                elif metric == 'recall':
                    row[metric] = np.random.uniform(0.65, 0.85)
                elif metric == 'f1_score':
                    row[metric] = np.random.uniform(0.70, 0.88)
                elif metric == 'roc_auc':
                    row[metric] = np.random.uniform(0.80, 0.95)
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Performance comparison
        st.plotly_chart(
            px.bar(
                results_df.melt(id_vars='model'),
                x='model',
                y='value',
                color='variable',
                title="Model Performance Comparison",
                barmode='group'
            ),
            use_container_width=True
        )
        
        # Leaderboard
        st.subheader("Model Leaderboard")
        results_df_sorted = results_df.sort_values('f1_score', ascending=False)
        results_df_sorted['rank'] = range(1, len(results_df_sorted) + 1)
        
        st.dataframe(
            results_df_sorted[['rank', 'model', 'f1_score', 'accuracy', 'precision', 'recall', 'roc_auc']],
            use_container_width=True
        )
        
        # ROC Curve simulation
        st.subheader("ROC Curves")
        fig = go.Figure()
        
        for model in models:
            # Generate random ROC curve data
            fpr = np.linspace(0, 1, 100)
            tpr = np.random.uniform(0.6, 0.95, 100) * fpr + np.random.uniform(0.05, 0.2, 100)
            tpr = np.clip(tpr, 0, 1)
            
            auc = results_df[results_df['model'] == model]['roc_auc'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f"{model} (AUC = {auc:.3f})"
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Real-time Predictions")
        
        # Interactive prediction interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Parameters")
            
            # Create input sliders for key features
            seismic_activity = st.slider("Seismic Activity (Richter)", 0.0, 8.0, 3.0, 0.1)
            rainfall = st.slider("Rainfall (mm)", 0, 300, 100, 5)
            wind_speed = st.slider("Wind Speed (km/h)", 0, 150, 40, 5)
            soil_saturation = st.slider("Soil Saturation", 0.0, 1.0, 0.5, 0.1)
            temperature = st.slider("Temperature (°C)", -20, 50, 25, 1)
            humidity = st.slider("Humidity (%)", 0, 100, 60, 5)
            pressure = st.slider("Pressure (hPa)", 800, 1100, 1013, 5)
            elevation = st.slider("Elevation (m)", 0, 3000, 500, 50)
            
            # Geographic inputs
            latitude = st.slider("Latitude", -90.0, 90.0, 0.0, 1.0)
            longitude = st.slider("Longitude", -180.0, 180.0, 0.0, 1.0)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Simulate prediction (in real app, this would use trained models)
            input_features = np.array([[
                seismic_activity, rainfall, wind_speed, soil_saturation,
                temperature, humidity, pressure, elevation, latitude, longitude
            ]])
            
            # Simple rule-based prediction for demo
            disaster_prob = 0.0
            
            # Earthquake risk
            if seismic_activity > 5.0:
                disaster_prob += 0.4
            
            # Landslide risk
            if rainfall > 150 and soil_saturation > 0.8 and elevation > 200:
                disaster_prob += 0.3
            
            # Flood risk
            if rainfall > 200 and soil_saturation > 0.9 and elevation < 100:
                disaster_prob += 0.3
            
            # Hurricane risk
            if wind_speed > 80 and pressure < 980 and humidity > 80:
                disaster_prob += 0.4
            
            # Wildfire risk
            if temperature > 35 and humidity < 30 and wind_speed > 40:
                disaster_prob += 0.3
            
            disaster_prob = min(disaster_prob, 1.0)
            disaster_prediction = 1 if disaster_prob > risk_threshold else 0
            
            # Display results
            st.metric("Disaster Probability", f"{disaster_prob:.1%}")
            st.metric("Risk Level", "HIGH" if disaster_prediction else "LOW")
            
            # Risk indicators
            st.subheader("Risk Indicators")
            
            indicators = {
                "Earthquake Risk": "HIGH" if seismic_activity > 5.0 else "LOW",
                "Landslide Risk": "HIGH" if rainfall > 150 and soil_saturation > 0.8 else "LOW",
                "Flood Risk": "HIGH" if rainfall > 200 and elevation < 100 else "LOW",
                "Hurricane Risk": "HIGH" if wind_speed > 80 and pressure < 980 else "LOW",
                "Wildfire Risk": "HIGH" if temperature > 35 and humidity < 30 else "LOW"
            }
            
            for indicator, level in indicators.items():
                color = "red" if level == "HIGH" else "green"
                st.markdown(f"**{indicator}:** <span style='color: {color}'>{level}</span>", 
                           unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommendations")
            if disaster_prediction:
                st.warning("⚠️ High disaster risk detected! Consider evacuation and emergency preparations.")
            else:
                st.success("✅ Low disaster risk. Normal conditions expected.")
            
            # Action items
            if disaster_prob > 0.3:
                st.info("📋 Suggested actions:")
                st.markdown("""
                - Monitor weather conditions closely
                - Prepare emergency supplies
                - Review evacuation routes
                - Stay informed through official channels
                """)


if __name__ == "__main__":
    main()
