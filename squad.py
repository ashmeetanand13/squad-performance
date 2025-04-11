import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Import data loader
from data_loader import load_data as load_github_data

# Function to load and process data
@st.cache_data
def load_data():
    """
    Load and process the football data at team level
    
    Returns:
        DataFrame: Team-level statistics
    """
    # Use the GitHub data loader
    df = load_github_data()
    
    # Return the loaded data
    return df

# Function to compute team-level metrics
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
        # Group by squad, competition, and season
        grouped = df.groupby(['Squad', 'Competition', 'Season'])
        
        # Define metrics to aggregate
        team_metrics = {}
        
        # Basic team information
        squads = df['Squad'].unique()
        competitions = df['Competition'].unique()
        seasons = df['Season'].unique()
        
        # Create team metrics dictionary
        teams_data = []
        
        for (squad, competition, season), team_df in grouped:
            team_data = {
                'Squad': squad,
                'Competition': competition,
                'Season': season
            }
            
            # Attack metrics
            attack_metrics = {
                # Goal scoring
                'Goals': team_df['Performance Gls'].sum(),
                'Goals Per 90': team_df['Performance Gls'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Shots': team_df['Standard Sh'].sum(),
                'Shots Per 90': team_df['Standard Sh'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Shot on Target %': 100 * team_df['Standard SoT'].sum() / max(1, team_df['Standard Sh'].sum()),
                'Goals Per Shot': team_df['Performance Gls'].sum() / max(1, team_df['Standard Sh'].sum()),
                'xG': team_df['Expected xG'].sum(),
                'xG Per 90': team_df['Expected xG'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'G-xG': team_df['Performance Gls'].sum() - team_df['Expected xG'].sum(),
                'Key Passes': team_df['KP'].sum(),
                'Key Passes Per 90': team_df['KP'].sum() / max(1, team_df['Playing Time 90s'].sum()),
            }
            
            # Possession metrics
            possession_metrics = {
                'Touches': team_df['Touches Touches'].sum(),
                'Touches Per 90': team_df['Touches Touches'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Progressive Carries': team_df['Carries PrgC'].sum(),
                'Progressive Carries Per 90': team_df['Carries PrgC'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Progressive Passes': team_df['PrgP'].sum(),
                'Progressive Passes Per 90': team_df['PrgP'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Attacking Third Touches %': 100 * team_df['Touches Att 3rd'].sum() / max(1, team_df['Touches Touches'].sum()),
                'Box Touches %': 100 * team_df['Touches Att Pen'].sum() / max(1, team_df['Touches Touches'].sum()),
                'Pass Completion %': 100 * team_df['Total Cmp'].sum() / max(1, team_df['Total Att'].sum()),
            }
            
            # Defense metrics
            defense_metrics = {
                'Tackles': team_df['Tackles Tkl'].sum(),
                'Tackles Per 90': team_df['Tackles Tkl'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Interceptions': team_df['Int'].sum(),
                'Interceptions Per 90': team_df['Int'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Tackles + Interceptions': team_df['Tkl+Int'].sum(),
                'Tackles + Interceptions Per 90': team_df['Tkl+Int'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Blocks': team_df['Blocks Blocks'].sum(),
                'Blocks Per 90': team_df['Blocks Blocks'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Clearances': team_df['Clr'].sum(),
                'Clearances Per 90': team_df['Clr'].sum() / max(1, team_df['Playing Time 90s'].sum()),
                'Errors': team_df['Err'].sum(),
                'Errors Per 90': team_df['Err'].sum() / max(1, team_df['Playing Time 90s'].sum()),
            }
            
            # Combine all metrics
            team_data.update(attack_metrics)
            team_data.update(possession_metrics)
            team_data.update(defense_metrics)
            
            teams_data.append(team_data)
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_text.text("Creating team metrics dataframe...")
        
        # Create DataFrame from the team metrics
        teams_df = pd.DataFrame(teams_data)
        
        status_text.text("Team metrics processing complete!")
        
        return teams_df
    
    except Exception as e:
        st.error(f"Error computing team metrics: {str(e)}")
        return None

# Function to normalize metrics within each competition
def normalize_metrics(teams_df):
    """
    Normalize team metrics within each competition to a 0-1 scale
    
    Args:
        teams_df: DataFrame with team metrics
        
    Returns:
        DataFrame: Teams with normalized metrics
    """
    if teams_df is None or teams_df.empty:
        return None
    
    status_text.text("Normalizing metrics within each competition...")
    
    # Create a copy of the DataFrame
    normalized_df = teams_df.copy()
    
    # List of metrics to normalize
    # Exclude identification columns like 'Squad', 'Competition', 'Season'
    # Also exclude percentage metrics which are already normalized
    exclude_cols = ['Squad', 'Competition', 'Season', 'Shot on Target %', 'Attacking Third Touches %', 'Box Touches %', 'Pass Completion %']
    invert_cols = ['Errors', 'Errors Per 90']  # Lower is better for these
    
    # Get all numeric columns
    numeric_cols = normalized_df.select_dtypes(include=['number']).columns
    metrics_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Show progress
    total_competitions = len(normalized_df['Competition'].unique())
    progress_bar = progress_placeholder.progress(0)
    
    # Normalize metrics within each competition
    for i, competition in enumerate(normalized_df['Competition'].unique()):
        # Update progress
        progress_percent = min(100, int((i / total_competitions) * 100))
        progress_bar.progress(progress_percent / 100)
        status_text.text(f"Normalizing metrics for competition {i+1}/{total_competitions}: {competition}")
        
        comp_mask = normalized_df['Competition'] == competition
        
        for col in metrics_to_normalize:
            col_min = normalized_df.loc[comp_mask, col].min()
            col_max = normalized_df.loc[comp_mask, col].max()
            
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
    
    # Clear progress indicators
    progress_placeholder.empty()
    status_text.text("Normalization complete!")
    
    return normalized_df

# Function to calculate similarity between two teams
def calculate_similarity(team1_data, team2_data):
    """
    Calculate similarity score between two teams based on their normalized metrics
    
    Args:
        team1_data: Series with team 1 metrics
        team2_data: Series with team 2 metrics
        
    Returns:
        float: Similarity score (0-1)
    """
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

# Function to create bar charts for team metrics
def create_team_metrics_chart(team_data, metric_category, normalized=True):
    """
    Create bar charts for team metrics
    
    Args:
        team_data: Series with team metrics
        metric_category: Category of metrics to display (Attack, Possession, Defense)
        normalized: Whether to use normalized metrics
        
    Returns:
        plotly figure
    """
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

# Function to create comparison bar charts for two teams
def create_comparison_chart(team1_data, team2_data, metric_category, normalized=True):
    """
    Create comparison bar charts for two teams
    
    Args:
        team1_data: Series with team 1 metrics
        team2_data: Series with team 2 metrics
        metric_category: Category of metrics to display
        normalized: Whether to use normalized metrics
        
    Returns:
        plotly figure
    """
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

# Function to create radar chart for team comparison
def create_radar_chart(team1_data, team2_data):
    """
    Create radar chart comparing two teams across key metrics
    
    Args:
        team1_data: Series with team 1 metrics
        team2_data: Series with team 2 metrics
        
    Returns:
        plotly figure
    """
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

# Main function
def main():
    # Title and introduction
    st.markdown('<p class="main-header">⚽ Football Team Playing Style Analyzer</p>', unsafe_allow_html=True)
    
    # Initialization of progress placeholders (used in other functions)
    global progress_placeholder, status_text
    progress_placeholder = st.empty()
    status_text = st.empty()
    
    # Load and process data
    df = load_data()
    
    if df is not None:
        # Show loading message
        with st.spinner("Computing team metrics..."):
            # Compute team-level metrics
            teams_df = compute_team_metrics(df)
            
            # Normalize metrics
            normalized_teams_df = normalize_metrics(teams_df)
        
        if normalized_teams_df is not None:
            # Optional data caching info
            if "normalized_teams_df" in st.session_state:
                st.success("Using cached data from previous run for faster performance")
            else:
                st.session_state["normalized_teams_df"] = normalized_teams_df
                
            # Clear any lingering status messages
            status_text.empty()
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
                            team_rank = sorted_df.index[sorted_df['Squad'] == selected_team].tolist()
                            
                            if team_rank:
                                rank = team_rank[0] + 1  # +1 because index is 0-based
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
                            <div class="similarity-value">{similarity*100:.2f}%</div>
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
            st.error("Error processing team metrics.")
    
    else:
        st.error("Error loading data. Please check if the data file is available.")

# Run the main function
if __name__ == "__main__":
    main()
