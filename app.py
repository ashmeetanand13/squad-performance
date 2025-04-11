import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import StringIO
import traceback

# Set page config
st.set_page_config(
    page_title="Football Team Playing Style Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    .similarity-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background-color: #f8f9fa;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        border: 3px solid #1E3A8A;
    }
    .similarity-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .similarity-label {
        font-size: 0.9rem;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """
    Load football data directly from GitHub with improved error handling
    
    Returns:
        DataFrame: Player-level data or None if loading fails
    """
    # The raw GitHub URL for your data
    GITHUB_RAW_URL = 'https://raw.githubusercontent.com/ashmeetanand13/squad-performance/main/df_clean.csv'
    
    try:
        # Show loading status
        with st.spinner("Loading data from GitHub..."):
            # Fetch data from GitHub
            response = requests.get(GITHUB_RAW_URL, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse CSV data with more robust settings
            content = StringIO(response.text)
            
            # Try different parsers and settings
            try:
                # First try: with C engine but skip bad lines
                df = pd.read_csv(
                    content, 
                    low_memory=False,
                    on_bad_lines='skip',
                    engine='c'  # Use C engine which supports low_memory
                )
                if df.shape[0] > 0:
                    st.info(f"Successfully loaded real data, skipping some malformed lines. Shape: {df.shape}")
                else:
                    raise ValueError("No valid rows found in CSV")
                    
            except Exception as e1:
                st.warning(f"First parsing attempt failed: {str(e1)}")
                
                # Reset file pointer to beginning
                content.seek(0)
                
                try:
                    # Second try: with Python engine (no low_memory)
                    df = pd.read_csv(
                        content, 
                        on_bad_lines='skip',
                        engine='python'  # Python engine doesn't support low_memory
                    )
                    st.warning("CSV had some formatting issues. Some rows may have been skipped.")
                except Exception as e2:
                    st.error(f"Could not parse CSV file: {str(e2)}")
                    return None
            
            # Basic data cleaning
            if 'Rk' in df.columns:
                df = df.drop('Rk', axis=1)
            
            # Log success and return
            st.success(f"Successfully loaded real data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from GitHub: {str(e)}")
        
        # Detailed error information to help debugging
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response status code: {e.response.status_code}")
            st.error(f"Response text: {e.response.text[:500]}...")
        
        return None
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return None

@st.cache_data
def compute_team_metrics(df):
    """
    Aggregate player-level data to team-level metrics
    
    Args:
        df: DataFrame containing player-level data
        
    Returns:
        DataFrame: Aggregated team-level metrics
    """
    if df is None:
        return None
    
    try:
        # Check if required columns exist
        required_columns = ['Squad', 'Competition', 'Season']
        if not all(col in df.columns for col in required_columns):
            st.warning("Some required columns are missing. Looking for alternative column names...")
            
            # Map of possible column names
            column_mapping = {
                'Squad': ['Squad', 'Team', 'Club'],
                'Competition': ['Competition', 'Comp', 'League'],
                'Season': ['Season', 'Year', 'Season_Year']
            }
            
            # Try to find alternative column names
            for required, alternatives in column_mapping.items():
                if required not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            df = df.rename(columns={alt: required})
                            st.info(f"Renamed column '{alt}' to '{required}'")
                            break
        
        # Check again if we have all required columns
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {missing}")
            return None
        
        # Group by team, competition, and season
        grouped = df.groupby(['Squad', 'Competition', 'Season'])
        
        # Create team metrics dictionary
        teams_data = []
        
        # Based on your CSV info, these are some relevant column mappings
        metric_mappings = {
            'Goals': ['Performance Gls', 'Standard Gls', 'Performance G-PK'],
            'xG': ['Expected xG', 'xG'],
            'Shots': ['Standard Sh'],
            'Shots on Target': ['Standard SoT'],
            'Playing Time': ['Playing Time 90s', '90s'],
            'Touches': ['Touches Touches'],
            'Progressive Passes': ['PrgP', 'Progression PrgP'],
            'Progressive Carries': ['Progression PrgC', 'Carries PrgC'],
            'Tackles': ['Tackles Tkl'],
            'Interceptions': ['Int'],
            'Blocks': ['Blocks Blocks', 'Blocks'],
            'Clearances': ['Clr'],
            'Key Passes': ['KP'],
            'Pass Completion': ['Total Cmp%'],
            'Attacking Third Touches': ['Touches Att 3rd'],
            'Box Touches': ['Touches Att Pen'],
            'Errors': ['Err']
        }
        
        # Process each team
        for (squad, competition, season), team_df in grouped:
            # Create team data dictionary with identification
            team_data = {
                'Squad': squad,
                'Competition': competition,
                'Season': season
            }
            
            # For each metric, try to find and sum the corresponding column
            for metric, possible_columns in metric_mappings.items():
                for col in possible_columns:
                    if col in team_df.columns:
                        # For percentage columns, take weighted average
                        if 'Cmp%' in col:
                            # Find the corresponding attempts column
                            att_col = col.replace('Cmp%', 'Att')
                            if att_col in team_df.columns:
                                completions = (team_df[col] * team_df[att_col]).sum()
                                attempts = team_df[att_col].sum()
                                team_data[f'Pass Completion %'] = 100 * completions / max(1, attempts)
                        # For regular columns, sum the values
                        else:
                            team_data[metric] = team_df[col].sum()
                        break
            
            # Calculate per 90 metrics
            playing_time = 0
            for time_col in ['Playing Time 90s', '90s']:
                if time_col in team_df.columns:
                    playing_time = team_df[time_col].sum()
                    break
            
            if playing_time > 0:
                # Create per 90 metrics
                for base_metric, per90_metric in [
                    ('Goals', 'Goals Per 90'),
                    ('xG', 'xG Per 90'),
                    ('Shots', 'Shots Per 90'),
                    ('Key Passes', 'Key Passes Per 90'),
                    ('Touches', 'Touches Per 90'),
                    ('Progressive Passes', 'Progressive Passes Per 90'),
                    ('Progressive Carries', 'Progressive Carries Per 90'),
                    ('Tackles', 'Tackles Per 90'),
                    ('Interceptions', 'Interceptions Per 90'),
                    ('Blocks', 'Blocks Per 90'),
                    ('Clearances', 'Clearances Per 90'),
                    ('Errors', 'Errors Per 90')
                ]:
                    if base_metric in team_data:
                        team_data[per90_metric] = team_data[base_metric] / playing_time
            
            # Calculate percentages
            if 'Shots' in team_data and 'Shots on Target' in team_data and team_data['Shots'] > 0:
                team_data['Shot on Target %'] = 100 * team_data['Shots on Target'] / team_data['Shots']
            
            if 'Goals' in team_data and 'Shots' in team_data and team_data['Shots'] > 0:
                team_data['Goals Per Shot'] = team_data['Goals'] / team_data['Shots']
            
            if 'Attacking Third Touches' in team_data and 'Touches' in team_data and team_data['Touches'] > 0:
                team_data['Attacking Third Touches %'] = 100 * team_data['Attacking Third Touches'] / team_data['Touches']
            
            if 'Box Touches' in team_data and 'Touches' in team_data and team_data['Touches'] > 0:
                team_data['Box Touches %'] = 100 * team_data['Box Touches'] / team_data['Touches']
            
            # Calculate derived metrics
            if 'Goals' in team_data and 'xG' in team_data:
                team_data['G-xG'] = team_data['Goals'] - team_data['xG']
            
            if 'Tackles' in team_data and 'Interceptions' in team_data:
                team_data['Tackles + Interceptions'] = team_data['Tackles'] + team_data['Interceptions']
                if playing_time > 0:
                    team_data['Tackles + Interceptions Per 90'] = team_data['Tackles + Interceptions'] / playing_time
            
            # Add team data to list
            teams_data.append(team_data)
        
        # Create DataFrame from the team metrics
        teams_df = pd.DataFrame(teams_data)
        
        # Display information about successful conversion
        st.info(f"Successfully aggregated data to {teams_df.shape[0]} teams")
        return teams_df
    
    except Exception as e:
        st.error(f"Error computing team metrics: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Sample data generation
@st.cache_data
def load_sample_data():
    """
    Create sample data for football team analysis
    
    Returns:
        DataFrame: Sample team-level data
    """
    # Create sample teams data
    teams_data = []
    
    # Generate data for major leagues
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
    
    for league in leagues:
        # Generate team names based on league
        if league == 'Premier League':
            teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 
                     'Manchester United', 'Newcastle', 'West Ham', 'Leicester', 'Brighton']
        elif league == 'La Liga':
            teams = ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Real Betis',
                     'Real Sociedad', 'Villarreal', 'Athletic Bilbao', 'Valencia', 'Osasuna']
        elif league == 'Bundesliga':
            teams = ['Bayern Munich', 'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 
                     'Union Berlin', 'Freiburg', 'Cologne', 'Mainz', 'Hoffenheim', 'Borussia Monchengladbach']
        elif league == 'Serie A':
            teams = ['AC Milan', 'Inter Milan', 'Napoli', 'Juventus', 'Lazio', 
                     'Roma', 'Fiorentina', 'Atalanta', 'Verona', 'Torino']
        else:  # Ligue 1
            teams = ['PSG', 'Marseille', 'Monaco', 'Rennes', 'Nice', 
                     'Strasbourg', 'Lens', 'Lyon', 'Nantes', 'Lille']
        
        # Generate data for each team
        for team in teams:
            # Base metrics with randomization
            goals = np.random.randint(40, 100)
            shots = np.random.randint(400, 700)
            shots_on_target = np.random.randint(int(shots * 0.3), int(shots * 0.5))
            touches = np.random.randint(15000, 25000)
            passes = np.random.randint(10000, 18000)
            tackles = np.random.randint(400, 700)
            interceptions = np.random.randint(300, 600)
            blocks = np.random.randint(200, 400)
            
            # Calculate derived metrics
            played_90s = 38 * 11  # Approx. for a full season of starters
            
            team_data = {
                'Squad': team,
                'Competition': league,
                'Season': '2022-23',
                
                # Attack metrics
                'Goals': goals,
                'Goals Per 90': goals / played_90s,
                'Shots': shots,
                'Shots Per 90': shots / played_90s,
                'Shot on Target %': 100 * shots_on_target / shots,
                'Goals Per Shot': goals / shots,
                'xG': goals * (0.9 + np.random.random() * 0.2),  # Randomize around goals
                'xG Per 90': (goals * (0.9 + np.random.random() * 0.2)) / played_90s,
                'G-xG': goals - (goals * (0.9 + np.random.random() * 0.2)),
                'Key Passes': np.random.randint(300, 500),
                'Key Passes Per 90': np.random.randint(300, 500) / played_90s,
                
                # Possession metrics
                'Touches': touches,
                'Touches Per 90': touches / played_90s,
                'Progressive Carries': np.random.randint(800, 1200),
                'Progressive Carries Per 90': np.random.randint(800, 1200) / played_90s,
                'Progressive Passes': np.random.randint(800, 1200),
                'Progressive Passes Per 90': np.random.randint(800, 1200) / played_90s,
                'Attacking Third Touches %': np.random.randint(20, 40),
                'Box Touches %': np.random.randint(5, 15),
                'Pass Completion %': np.random.randint(75, 90),
                
                # Defense metrics
                'Tackles': tackles,
                'Tackles Per 90': tackles / played_90s,
                'Interceptions': interceptions,
                'Interceptions Per 90': interceptions / played_90s,
                'Tackles + Interceptions': tackles + interceptions,
                'Tackles + Interceptions Per 90': (tackles + interceptions) / played_90s,
                'Blocks': blocks,
                'Blocks Per 90': blocks / played_90s,
                'Clearances': np.random.randint(500, 800),
                'Clearances Per 90': np.random.randint(500, 800) / played_90s,
                'Errors': np.random.randint(10, 30),
                'Errors Per 90': np.random.randint(10, 30) / played_90s,
            }
            
            teams_data.append(team_data)
    
    # Create DataFrame from the sample data
    teams_df = pd.DataFrame(teams_data)
    
    # Normalize metrics within each competition
    normalized_teams_df = normalize_metrics(teams_df)
    
    st.success("Using sample data with 50 teams across 5 major leagues")
    return normalized_teams_df

@st.cache_data
def load_and_process_data():
    """
    Load and process the football data
    
    Returns:
        DataFrame: Processed team-level data
    """
    # First, try to load real data
    df = load_real_data()
    
    if df is not None:
        # Process the real data - aggregate from player to team level
        try:
            st.info("Processing player data to team-level metrics...")
            # Compute team metrics from player data
            teams_df = compute_team_metrics(df)
            
            if teams_df is not None:
                # Normalize the metrics
                normalized_teams_df = normalize_metrics(teams_df)
                st.success("Successfully processed real data!")
                return normalized_teams_df
            else:
                raise Exception("Failed to compute team metrics")
        except Exception as e:
            st.error(f"Error processing real data: {str(e)}")
            st.warning("Falling back to sample data")
            return load_sample_data()
    else:
        # If real data loading fails, use sample data
        st.warning("Using sample data because real data could not be loaded")
        return load_sample_data()

@st.cache_data
def calculate_similarity(team1_data, team2_data):
    """Calculate similarity score between two teams based on their normalized metrics"""
    # Get all normalized metrics
    normalized_cols = [col for col in team1_data.index if col.startswith('Normalized')]
    
    # Calculate Euclidean distance
    squared_diff_sum = 0
    valid_metrics = 0
    
    for col in normalized_cols:
        if col in team1_data and col in team2_data:
            val1 = team1_data[col]
            val2 = team2_data[col]
            
            # Skip if either value is NaN
            if pd.notna(val1) and pd.notna(val2):
                squared_diff_sum += (val1 - val2) ** 2
                valid_metrics += 1
    
    if valid_metrics == 0:
        return 0
    
    # Calculate similarity (inverse of normalized distance)
    similarity = 1 - (np.sqrt(squared_diff_sum / valid_metrics) / np.sqrt(2))
    
    return max(0, min(1, similarity))  # Ensure between 0 and 1

@st.cache_data
def create_team_metrics_chart(team_data, metric_category, normalized=True):
    """Create bar charts for team metrics"""
    # Define metrics for each category
    metrics_by_category = {
        'Attack': [
            'Goals Per 90', 'Shots Per 90', 'Shot on Target %', 
            'Goals Per Shot', 'xG Per 90', 'G-xG', 'Key Passes Per 90'
        ],
        'Possession': [
            'Touches Per 90', 'Progressive Carries Per 90', 'Progressive Passes Per 90',
            'Attacking Third Touches %', 'Box Touches %', 'Pass Completion %'
        ],
        'Defense': [
            'Tackles Per 90', 'Interceptions Per 90', 'Tackles + Interceptions Per 90',
            'Blocks Per 90', 'Clearances Per 90', 'Errors Per 90'
        ]
    }
    
    # Get metrics for the selected category
    selected_metrics = metrics_by_category.get(metric_category, [])
    
    if normalized:
        # Use normalized metrics
        selected_metrics = [f'Normalized {m}' for m in selected_metrics]
        y_range = [0, 1]
        title_suffix = "Normalized (0-1 scale)"
    else:
        # Use raw metrics
        y_range = None
        title_suffix = "Raw Values"
    
    # Filter metrics that exist in the data
    available_metrics = [m for m in selected_metrics if m in team_data.index]
    
    if not available_metrics:
        return None
    
    # Extract values
    values = team_data[available_metrics]
    labels = [m.replace('Normalized ', '') for m in available_metrics]
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=['#1E3A8A' if normalized else '#3B82F6'],
        name=team_data['Squad'],
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric_category} Metrics - {title_suffix}",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        yaxis=dict(range=y_range),
        template="plotly_white"
    )
    
    return fig

@st.cache_data
def create_comparison_chart(team1_data, team2_data, metric_category, normalized=True):
    """Create comparison bar charts for two teams"""
    # Define metrics for each category
    metrics_by_category = {
        'Attack': [
            'Goals Per 90', 'Shots Per 90', 'Shot on Target %', 
            'Goals Per Shot', 'xG Per 90', 'G-xG', 'Key Passes Per 90'
        ],
        'Possession': [
            'Touches Per 90', 'Progressive Carries Per 90', 'Progressive Passes Per 90',
            'Attacking Third Touches %', 'Box Touches %', 'Pass Completion %'
        ],
        'Defense': [
            'Tackles Per 90', 'Interceptions Per 90', 'Tackles + Interceptions Per 90',
            'Blocks Per 90', 'Clearances Per 90', 'Errors Per 90'
        ]
    }
    
    # Get metrics for the selected category
    selected_metrics = metrics_by_category.get(metric_category, [])
    
    if normalized:
        # Use normalized metrics
        selected_metrics = [f'Normalized {m}' for m in selected_metrics]
        y_range = [0, 1]
        title_suffix = "Normalized (0-1 scale)"
    else:
        # Use raw metrics
        y_range = None
        title_suffix = "Raw Values"
    
    # Filter metrics that exist in both datasets
    available_metrics = [m for m in selected_metrics if m in team1_data.index and m in team2_data.index]
    
    if not available_metrics:
        return None
    
    # Extract values
    values1 = team1_data[available_metrics]
    values2 = team2_data[available_metrics]
    labels = [m.replace('Normalized ', '') for m in available_metrics]
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars for team 1
    fig.add_trace(go.Bar(
        x=labels,
        y=values1,
        marker_color='#1E3A8A',
        name=team1_data['Squad'],
        text=[f"{v:.2f}" for v in values1],
        textposition='auto',
    ))
    
    # Add bars for team 2
    fig.add_trace(go.Bar(
        x=labels,
        y=values2,
        marker_color='#DC2626',
        name=team2_data['Squad'],
        text=[f"{v:.2f}" for v in values2],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric_category} Metrics Comparison - {title_suffix}",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode='group',
        height=500,
        yaxis=dict(range=y_range),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@st.cache_data
def create_radar_chart(team1_data, team2_data):
    """Create radar chart comparing two teams across key metrics"""
    # Select key normalized metrics for radar chart
    radar_metrics = [
        'Normalized Goals Per 90', 
        'Normalized Shots Per 90',
        'Normalized Shot on Target %',
        'Normalized xG Per 90',
        'Normalized Touches Per 90',
        'Normalized Progressive Carries Per 90',
        'Normalized Progressive Passes Per 90',
        'Normalized Pass Completion %',
        'Normalized Tackles Per 90',
        'Normalized Interceptions Per 90',
        'Normalized Blocks Per 90'
    ]
    
    # Filter metrics that exist in both datasets
    available_metrics = [m for m in radar_metrics if m in team1_data.index and m in team2_data.index]
    
    if not available_metrics:
        return None
    
    # Extract values
    values1 = team1_data[available_metrics].values
    values2 = team2_data[available_metrics].values
    labels = [m.replace('Normalized ', '') for m in available_metrics]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add radar for team 1
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=labels,
        fill='toself',
        name=team1_data['Squad'],
        line_color='#1E3A8A',
        fillcolor='rgba(30, 58, 138, 0.3)'
    ))
    
    # Add radar for team 2
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=labels,
        fill='toself',
        name=team2_data['Squad'],
        line_color='#DC2626',
        fillcolor='rgba(220, 38, 38, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Team Style Comparison",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=600,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@st.cache_data
def normalize_metrics(teams_df):
    """
    Normalize team metrics within each competition to a 0-1 scale
    
    Args:
        teams_df: DataFrame with team metrics
        
    Returns:
        DataFrame: Teams with normalized metrics
    """
    # Create a copy of the DataFrame
    normalized_df = teams_df.copy()
    
    # List of metrics to normalize
    exclude_cols = ['Squad', 'Competition', 'Season', 'Shot on Target %', 'Attacking Third Touches %', 'Box Touches %', 'Pass Completion %']
    invert_cols = ['Errors', 'Errors Per 90']  # Lower is better for these
    
    # Get all numeric columns
    numeric_cols = normalized_df.select_dtypes(include=['number']).columns
    metrics_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Normalize metrics within each competition
    for competition in normalized_df['Competition'].unique():
        comp_mask = normalized_df['Competition'] == competition
        
        for col in metrics_to_normalize:
            # Skip if column doesn't exist
            if col not in normalized_df.columns:
                continue
                
            col_values = normalized_df.loc[comp_mask, col]
            
            # Skip if no valid values or all values are the same
            if col_values.isna().all() or col_values.nunique() <= 1:
                normalized_df.loc[comp_mask, f'Normalized {col}'] = 0.5
                continue
                
            col_min = col_values.min()
            col_max = col_values.max()
            
            # Skip if min equals max (no variation)
            if col_max > col_min:
                if col in invert_cols:
                    # For metrics where lower is better
                    normalized_df.loc[comp_mask, f'Normalized {col}'] = 1 - (normalized_df.loc[comp_mask, col] - col_min) / (col_max - col_min)
                else:
                    # For metrics where higher is better
                    normalized_df.loc[comp_mask, f'Normalized {col}'] = (normalized_df.loc[comp_mask, col] - col_min) / (col_max - col_min)
            else:
                # If all values are the same, set normalized value to 0.5
                normalized_df.loc[comp_mask, f'Normalized {col}'] = 0.5
    
    # For percentage metrics, divide by 100 to get 0-1 scale
    for col in ['Shot on Target %', 'Attacking Third Touches %', 'Box Touches %', 'Pass Completion %']:
        if col in normalized_df.columns:
            normalized_df[f'Normalized {col}'] = normalized_df[col] / 100
    
    return normalized_df

def main():
    """Main function to run the Streamlit application"""
    # Title and introduction
    st.markdown('<p class="main-header">⚽ Football Team Playing Style Analyzer</p>', unsafe_allow_html=True)
    
    # Add last updated timestamp
    st.markdown(f"<p style='text-align: right; color: #888;'>Last updated: {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
    
    # Initialize session state vars if needed
    if 'normalized_teams_df' not in st.session_state:
        st.session_state['normalized_teams_df'] = None
    
    # Add a toggle to force sample data for testing
    use_sample_data = st.sidebar.checkbox("Use sample data (for testing)", value=False)
    
    # Load and process data
    with st.spinner("Loading data..."):
        if use_sample_data:
            # Use sample data if checkbox is selected
            normalized_teams_df = load_sample_data()
            st.info("Using sample data as requested")
        else:
            # Try to load real data first, fall back to sample if needed
            normalized_teams_df = load_and_process_data()
            
        # Store in session state
        st.session_state['normalized_teams_df'] = normalized_teams_df
    
    if normalized_teams_df is not None:
        # Sidebar for filtering and navigation
        st.sidebar.markdown("## Navigation")
        app_mode = st.sidebar.radio("Select Mode", ["Single Team Analysis", "Team Comparison"])
        
        # Get unique competitions, seasons, and teams
        competitions = sorted(normalized_teams_df['Competition'].unique())
        seasons = sorted(normalized_teams_df['Season'].unique(), reverse=True)
        
        # Filter by season
        selected_season = st.sidebar.selectbox("Select Season", seasons)
        season_mask = normalized_teams_df['Season'] == selected_season
        
        if app_mode == "Single Team Analysis":
            # Single team analysis mode
            st.markdown('<p class="sub-header">Single Team Analysis</p>', unsafe_allow_html=True)
            
            # Filter selectors
            col1, col2 = st.columns(2)
            
            with col1:
                # Competition selector
                selected_competition = st.selectbox(
                    "Select Competition", 
                    competitions,
                    index=0 if competitions else None
                )
                
                # Update teams based on competition
                comp_mask = normalized_teams_df['Competition'] == selected_competition
                filtered_df = normalized_teams_df[comp_mask & season_mask]
                teams = sorted(filtered_df['Squad'].unique())
            
            with col2:
                # Team selector
                selected_team = st.selectbox(
                    "Select Team",
                    teams,
                    index=0 if teams else None
                )
            
            # Get team data
            team_mask = filtered_df['Squad'] == selected_team
            if not filtered_df[team_mask].empty:
                team_data = filtered_df[team_mask].iloc[0]
                
                # Option to show raw or normalized values
                show_normalized = st.checkbox("Show Normalized Values", value=True)
                
                # Display team metrics in categories
                tabs = st.tabs(["Attack", "Possession", "Defense"])
                
                # Attack tab
                with tabs[0]:
                    attack_chart = create_team_metrics_chart(team_data, "Attack", normalized=show_normalized)
                    if attack_chart:
                        st.plotly_chart(attack_chart, use_container_width=True)
                    else:
                        st.warning("No attack metrics available for this team.")
                
                # Possession tab
                with tabs[1]:
                    possession_chart = create_team_metrics_chart(team_data, "Possession", normalized=show_normalized)
                    if possession_chart:
                        st.plotly_chart(possession_chart, use_container_width=True)
                    else:
                        st.warning("No possession metrics available for this team.")
                
                # Defense tab
                with tabs[2]:
                    defense_chart = create_team_metrics_chart(team_data, "Defense", normalized=show_normalized)
                    if defense_chart:
                        st.plotly_chart(defense_chart, use_container_width=True)
                    else:
                        st.warning("No defense metrics available for this team.")
                
                # Team ranking in competition
                st.markdown('<p class="sub-header">Team Ranking in Competition</p>', unsafe_allow_html=True)
                
                # Get team's ranking for key metrics
                key_metrics = [
                    'Goals Per 90', 'xG Per 90', 'Shots Per 90', 
                    'Progressive Passes Per 90', 'Pass Completion %',
                    'Tackles + Interceptions Per 90'
                ]
                
                # Create ranking table
                ranking_data = []
                
                for metric in key_metrics:
                    if metric in filtered_df.columns:
                        # Sort teams by metric and get rank
                        sorted_df = filtered_df.sort_values(metric, ascending=False)
                        teams_count = len(sorted_df)
                        rank = sorted_df[sorted_df['Squad'] == selected_team].index[0] + 1  # +1 because index is 0-based
                        percentile = 100 * (teams_count - rank + 1) / teams_count
                        
                        ranking_data.append({
                            'Metric': metric,
                            'Value': team_data[metric],
                            'Rank': f"{rank}/{teams_count}",
                            'Percentile': f"{percentile:.1f}%"
                        })
                
                # Display ranking table
                if ranking_data:
                    ranking_df = pd.DataFrame(ranking_data)
                    st.dataframe(ranking_df, use_container_width=True)
                else:
                    st.warning("No ranking data available for this team.")
            
            else:
                st.warning("No data available for the selected team.")
        
        elif app_mode == "Team Comparison":
            # Team comparison mode
            st.markdown('<p class="sub-header">Team Comparison</p>', unsafe_allow_html=True)
            
            # First team selection
            st.markdown("### Select First Team")
            col1, col2 = st.columns(2)
            
            with col1:
                # Competition selector for team 1
                comp1 = st.selectbox(
                    "Competition (Team 1)", 
                    competitions,
                    key="comp1"
                )
                
                # Update teams based on competition
                comp1_mask = normalized_teams_df['Competition'] == comp1
                filtered_df1 = normalized_teams_df[comp1_mask & season_mask]
                teams1 = sorted(filtered_df1['Squad'].unique())
            
            with col2:
                # Team 1 selector
                team1 = st.selectbox(
                    "Team 1",
                    teams1,
                    key="team1"
                )
            
            # Second team selection
            st.markdown("### Select Second Team")
            col1, col2 = st.columns(2)
            
            with col1:
                # Competition selector for team 2
                comp2 = st.selectbox(
                    "Competition (Team 2)", 
                    competitions,
                    key="comp2"
                )
                
                # Update teams based on competition
                comp2_mask = normalized_teams_df['Competition'] == comp2
                filtered_df2 = normalized_teams_df[comp2_mask & season_mask]
                teams2 = sorted(filtered_df2['Squad'].unique())
            
            with col2:
                # Team 2 selector
                team2 = st.selectbox(
                    "Team 2",
                    teams2,
                    key="team2"
                )
            
            # Get team data
            team1_mask = filtered_df1['Squad'] == team1
            team2_mask = filtered_df2['Squad'] == team2
            
            if not filtered_df1[team1_mask].empty and not filtered_df2[team2_mask].empty:
                team1_data = filtered_df1[team1_mask].iloc[0]
                team2_data = filtered_df2[team2_mask].iloc[0]
                
                # Calculate similarity
                similarity = calculate_similarity(team1_data, team2_data)
                
                # Display similarity
                st.markdown(
                    f"""
                    <div class="similarity-circle">
                        <div class="similarity-value">{similarity*100:.1f}%</div>
                        <div class="similarity-label">Similarity</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Option to show raw or normalized values
                show_normalized = st.checkbox("Show Normalized Values", value=True)
                
                # Display comparison in categories
                tabs = st.tabs(["Overview", "Attack", "Possession", "Defense"])
                
                # Overview tab (radar chart)
                with tabs[0]:
                    radar_chart = create_radar_chart(team1_data, team2_data)
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
                    else:
                        st.warning("Not enough comparable metrics available for these teams.")
                
                # Attack tab
                with tabs[1]:
                    attack_chart = create_comparison_chart(team1_data, team2_data, "Attack", normalized=show_normalized)
                    if attack_chart:
                        st.plotly_chart(attack_chart, use_container_width=True)
                    else:
                        st.warning("No comparable attack metrics available for these teams.")
                
                # Possession tab
                with tabs[2]:
                    possession_chart = create_comparison_chart(team1_data, team2_data, "Possession", normalized=show_normalized)
                    if possession_chart:
                        st.plotly_chart(possession_chart, use_container_width=True)
                    else:
                        st.warning("No comparable possession metrics available for these teams.")
                
                # Defense tab
                with tabs[3]:
                    defense_chart = create_comparison_chart(team1_data, team2_data, "Defense", normalized=show_normalized)
                    if defense_chart:
                        st.plotly_chart(defense_chart, use_container_width=True)
                    else:
                        st.warning("No comparable defense metrics available for these teams.")
                
                # Key metrics comparison table
                st.markdown('<p class="sub-header">Key Metrics Comparison</p>', unsafe_allow_html=True)
                
                # Define key metrics to compare
                key_metrics = [
                    'Goals', 'Goals Per 90', 'xG', 'G-xG',
                    'Shots Per 90', 'Shot on Target %',
                    'Touches Per 90', 'Pass Completion %',
                    'Progressive Passes Per 90', 'Progressive Carries Per 90',
                    'Tackles + Interceptions Per 90'
                ]
                
                # Create comparison table
                comparison_data = []
                
                for metric in key_metrics:
                    if metric in team1_data.index and metric in team2_data.index:
                        val1 = team1_data[metric]
                        val2 = team2_data[metric]
                        
                        comparison_data.append({
                            'Metric': metric,
                            f'{team1}': val1,
                            f'{team2}': val2,
                            'Difference': val1 - val2
                        })
                
                # Display comparison table
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.warning("No comparable metrics available for these teams.")
            
            else:
                st.warning("Data missing for one or both selected teams.")
        
        # Add a section about data source and how to use
        st.markdown("---")
        st.markdown('<p class="sub-header">How to Use This Dashboard</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Single Team Analysis
        1. Select a competition and team to view their performance metrics
        2. Switch between attack, possession, and defense tabs to view different metric categories
        3. Toggle between normalized and raw values using the checkbox
        4. Check team rankings to see how they compare to other teams in the same competition
        
        ### Team Comparison
        1. Select two teams (from the same or different competitions)
        2. View their similarity score based on playing style
        3. Compare their metrics across different categories using the tabs
        4. Use the overview tab to see a radar chart showing strengths and weaknesses
        
        ### Metrics Explanation
        - **Attack Metrics**: Goals, shots, and chance creation statistics
        - **Possession Metrics**: Ball retention, progression, and passing metrics
        - **Defense Metrics**: Tackles, interceptions, blocks, and other defensive actions
        
        ### Normalization
        All metrics are normalized within each competition on a 0-1 scale where:
        - 0 = Lowest value in the competition
        - 1 = Highest value in the competition
        
        This allows for fair comparison of teams across different competitions with different playing styles and competitive levels.
        """)
    
    else:
        st.error("Error loading data. Please check if the data file is available.")

# Run the main function
if __name__ == "__main__":
    main()
