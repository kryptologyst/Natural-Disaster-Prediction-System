"""Map visualization utilities for disaster prediction."""

import numpy as np
import pandas as pd
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple, Optional
import json


class DisasterMapVisualizer:
    """Map visualization utilities for disaster prediction results."""
    
    def __init__(self):
        """Initialize the map visualizer."""
        self.default_center = [20, 0]  # Center of world map
        self.default_zoom = 2
        
    def create_risk_map(self, df: pd.DataFrame, risk_column: str = 'disaster_risk',
                       lat_column: str = 'latitude', lon_column: str = 'longitude',
                       title: str = "Disaster Risk Map") -> folium.Map:
        """Create a Folium map showing disaster risk locations.
        
        Args:
            df: DataFrame with location and risk data
            risk_column: Column name for risk values
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            title: Map title
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter').add_to(m)
        
        # Create color mapping based on risk levels
        def get_color(risk_value):
            if risk_value == 1:
                return 'red'
            else:
                return 'green'
        
        # Add markers for each location
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row[lat_column], row[lon_column]],
                radius=5,
                popup=f"Risk: {row[risk_column]}",
                color=get_color(row[risk_column]),
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_probability_map(self, df: pd.DataFrame, prob_column: str = 'disaster_probability',
                              lat_column: str = 'latitude', lon_column: str = 'longitude',
                              title: str = "Disaster Probability Map") -> folium.Map:
        """Create a map showing disaster probabilities with heatmap.
        
        Args:
            df: DataFrame with location and probability data
            prob_column: Column name for probability values
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            title: Map title
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Prepare data for heatmap
        heat_data = [[row[lat_column], row[lon_column], row[prob_column]] 
                    for idx, row in df.iterrows()]
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            name='Disaster Probability Heatmap',
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Add individual markers with probability info
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row[lat_column], row[lon_column]],
                radius=3,
                popup=f"Probability: {row[prob_column]:.3f}",
                color='black',
                fill=True,
                fillOpacity=0.5
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_interactive_plotly_map(self, df: pd.DataFrame, 
                                    risk_column: str = 'disaster_risk',
                                    prob_column: str = 'disaster_probability',
                                    lat_column: str = 'latitude', 
                                    lon_column: str = 'longitude',
                                    title: str = "Interactive Disaster Risk Map") -> go.Figure:
        """Create an interactive Plotly map.
        
        Args:
            df: DataFrame with location and risk data
            risk_column: Column name for risk values
            prob_column: Column name for probability values
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            title: Map title
            
        Returns:
            Plotly figure object
        """
        # Create scatter mapbox
        fig = go.Figure()
        
        # Add traces for different risk levels
        risk_0 = df[df[risk_column] == 0]
        risk_1 = df[df[risk_column] == 1]
        
        if not risk_0.empty:
            fig.add_trace(go.Scattermapbox(
                lat=risk_0[lat_column],
                lon=risk_0[lon_column],
                mode='markers',
                marker=dict(
                    size=8,
                    color='green',
                    opacity=0.7
                ),
                text=[f"Risk: Low<br>Probability: {p:.3f}" for p in risk_0[prob_column]],
                hovertemplate='%{text}<extra></extra>',
                name='Low Risk'
            ))
        
        if not risk_1.empty:
            fig.add_trace(go.Scattermapbox(
                lat=risk_1[lat_column],
                lon=risk_1[lon_column],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    opacity=0.7
                ),
                text=[f"Risk: High<br>Probability: {p:.3f}" for p in risk_1[prob_column]],
                hovertemplate='%{text}<extra></extra>',
                name='High Risk'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=20, lon=0),
                zoom=2
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_choropleth_map(self, df: pd.DataFrame, 
                            value_column: str = 'disaster_probability',
                            location_column: str = 'country',
                            title: str = "Disaster Risk by Country") -> go.Figure:
        """Create a choropleth map showing disaster risk by country.
        
        Args:
            df: DataFrame with country and risk data
            value_column: Column name for risk values
            location_column: Column name for country names
            title: Map title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Choropleth(
            locations=df[location_column],
            z=df[value_column],
            locationmode='country names',
            colorscale='Reds',
            colorbar_title="Disaster Risk",
            hovertemplate='Country: %{location}<br>Risk: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=600
        )
        
        return fig
    
    def create_risk_heatmap(self, df: pd.DataFrame, 
                          lat_column: str = 'latitude',
                          lon_column: str = 'longitude',
                          value_column: str = 'disaster_probability',
                          title: str = "Disaster Risk Heatmap") -> go.Figure:
        """Create a 2D heatmap of disaster risk.
        
        Args:
            df: DataFrame with location and risk data
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            value_column: Column name for risk values
            title: Map title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Densitymapbox(
            lat=df[lat_column],
            lon=df[lon_column],
            z=df[value_column],
            radius=10,
            colorscale='Reds',
            colorbar_title="Risk Level"
        ))
        
        fig.update_layout(
            title=title,
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=df[lat_column].mean(), lon=df[lon_column].mean()),
                zoom=4
            ),
            height=600
        )
        
        return fig
    
    def save_map(self, map_obj: folium.Map, filepath: str):
        """Save a Folium map to HTML file.
        
        Args:
            map_obj: Folium map object
            filepath: Path to save the HTML file
        """
        map_obj.save(filepath)
        print(f"Map saved to {filepath}")
    
    def create_map_dashboard(self, df: pd.DataFrame, 
                           save_path: str = "assets/disaster_map_dashboard.html") -> str:
        """Create a comprehensive map dashboard.
        
        Args:
            df: DataFrame with disaster prediction data
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create multiple map views
        risk_map = self.create_risk_map(df)
        prob_map = self.create_probability_map(df)
        plotly_map = self.create_interactive_plotly_map(df)
        
        # Save individual maps
        self.save_map(risk_map, "assets/risk_map.html")
        self.save_map(prob_map, "assets/probability_map.html")
        
        # Save Plotly map
        plotly_map.write_html("assets/interactive_map.html")
        
        # Create dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Disaster Prediction Map Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .map-container {{ margin: 20px 0; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                iframe {{ width: 100%; height: 500px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Disaster Prediction Map Dashboard</h1>
            <p>Interactive maps showing disaster risk predictions and probabilities.</p>
            
            <div class="map-container">
                <h2>Risk Classification Map</h2>
                <iframe src="risk_map.html"></iframe>
            </div>
            
            <div class="map-container">
                <h2>Probability Heatmap</h2>
                <iframe src="probability_map.html"></iframe>
            </div>
            
            <div class="map-container">
                <h2>Interactive Risk Map</h2>
                <iframe src="interactive_map.html"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Map dashboard saved to {save_path}")
        return save_path
