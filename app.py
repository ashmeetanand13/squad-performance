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

def clean_numeric_column(series):
    """Clean and convert a pandas series to numeric, handling common issues"""
    if series.dtype == 'object':
        # Handle common string issues in football data
        series = series.astype(str)
        series = series.str.replace(',', '')  # Remove commas
        series = series.str.replace('%', '')  # Remove percentage signs
        series = series.str.replace('€', '')  # Remove currency symbols
        series = series.str.replace('£', '')  # Remove currency symbols
        series = series.str.replace('$', '')  # Remove currency symbols
        series = series.str.strip()  # Remove whitespace
        
        # Replace common non-numeric values
        series = series.replace(['', '-', 'N/A', 'n/a', 'NaN', 'nan', 'null'], np.nan)
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(series, errors='coerce')

@st.cache_data
def load_real_data():
    """Load football data directly from GitHub with improved error handling"""
    GITHUB_RAW_URL = 'https://raw.githubusercontent.com/ashmeetanand13/squad-performance/main/df_clean.csv'
    
    try:
        with st.spinner("Loading data from GitHub..."):
            response = requests.get(GITHUB_RAW_URL, timeout=15)
            response.raise_for_status()
            
            content = StringIO(response.text)
            
            # Read CSV with robust settings
            df = pd.read_csv(
                content, 
                low_memory=False,
                on_bad_lines='skip',
                encoding='utf-8'
            )
            
            if df.shape[0] == 0:
                raise ValueError("No valid rows found in CSV")
            
            # Remove ranking column if it exists
            if 'Rk' in df.columns:
                df = df.drop('Rk', axis=1)
            
            st.success(f"Successfully loaded real data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def compute_team_metrics(df):
    """Aggregate player-level data to team-level metrics with robust data handling"""
    if df is None:
        return None
    
    try:
        # Check and fix required columns
        required_columns = ['Squad', 'Competition', 'Season']
        column_mapping = {
            'Squad': ['Squad', 'Team', 'Club'],
            'Competition': ['Competition', 'Comp', 'League'],
            'Season': ['Season', 'Year', 'Season_Year']
        }
        
        # Fix column names
        for required, alternatives in column_mapping.items():
            if required not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df = df.rename(columns={alt: required})
                        st.info(f"Renamed column '{alt}' to '{required}'")
                        break
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {missing}")
            return None
        
        # Clean and identify numeric columns
        numeric_columns = []
        for col in df.columns:
            if col not in required_columns:
                original_type = df[col].dtype
                df[col] = clean_numeric_column(df[col])
                if df[col].notna().any():  # If we have any valid numeric values
                    numeric_columns.append(col)
        
        st.info(f"Found {len(numeric_columns)} numeric columns to aggregate")
        
        # Group by team, competition, and season
        grouped = df.groupby(['Squad', 'Competition', 'Season'])
        teams_data = []
        
        # Process each team
        for (squad, competition, season), team_df in grouped:
            team_data = {
                'Squad': squad,
                'Competition': competition,
                'Season': season
            }
            
            # Calculate team totals for numeric columns
            for col in numeric_columns:
                if col in team_df.columns:
                    # For percentage columns, calculate weighted average
                    if any(keyword in col.lower() for keyword in ['%', 'pct', 'rate', 'completion']):
                        # Try to find corresponding total/attempt columns
                        total_cols = [c for c in numeric_columns if 
                                    any(keyword in c.lower() for keyword in ['total', 'att', 'attempt']) and
                                    col.replace('%', '').replace('Pct', '') in c]
                        
                        if total_cols:
                            weights = team_df[total_cols[0]].fillna(0)
                            if weights.sum() > 0:
                                weighted_avg = (team_df[col].fillna(0) * weights).sum() / weights.sum()
                                team_data[col] = weighted_avg
                            else:
                                team_data[col] = team_df[col].mean()
                        else:
                            team_data[col] = team_df[col].mean()
                    else:
                        # For regular metrics, sum the values
                        team_data[col] = team_df[col].sum()
            
            # Calculate per-90 metrics if we have playing time data
            time_columns = [col for col in numeric_columns if 
                          any(keyword in col.lower() for keyword in ['90', 'min', 'minutes', 'mp'])]
            
            if time_columns:
                total_90s = team_df[time_columns[0]].sum()
                if total_90s > 0:
                    # Create per-90 versions of key metrics
                    per_90_metrics = [col for col in numeric_columns if 
                                    any(keyword in col.lower() for keyword in 
                                        ['goal', 'shot', 'pass', 'tackle', 'interception', 'touch', 'carry'])]
                    
                    for metric in per_90_metrics:
                        if metric in team_data and '90' not in metric.lower():
                            team_data[f"{metric} Per 90"] = team_data[metric] / total_90s
            
            teams_data.append(team_data)
        
        # Create DataFrame
        teams_df = pd.DataFrame(teams_data)
        
        # Clean the final dataframe
        for col in teams_df.select_dtypes(include=['number']).columns:
            teams_df[col] = teams_df[col].fillna(0)
        
        st.info(f"Successfully aggregated data to {teams_df.shape[0]} teams")
        return teams_df
    
    except Exception as e:
        st.error(f"Error computing team metrics: {str(e)}")
        st.error(traceback.format_exc())
        return None



@st.cache_data
def load_and_process_data():
    """Load and process the football data"""
    df = load_real_data()
    
    if df is not None:
        try:
            st.info("Processing player data to team-level metrics...")
            teams_df = compute_team_metrics(df)
            
            if teams_df is not None:
                normalized_teams_df = normalize_metrics(teams_df)
                st.success("Successfully processed real data!")
                return normalized_teams_df
            else:
                st.error("Failed to compute team metrics from the data")
                return None
        except Exception as e:
            st.error(f"Error processing real data: {str(e)}")
            return None
    else:
        st.error("Could not load data from GitHub repository")
        return None

@st.cache_data
def normalize_metrics(teams_df):
    """Normalize team metrics within each competition to a 0-1 scale"""
    normalized_df = teams_df.copy()
    
    # Identify columns to normalize
    exclude_cols = ['Squad', 'Competition', 'Season']
    percentage_cols = [col for col in normalized_df.columns if '%' in col]
    
    numeric_cols = normalized_df.select_dtypes(include=['number']).columns
    metrics_to_normalize = [col for col in numeric_cols if col not in exclude_cols and col not in percentage_cols]
    
    # Normalize within each competition
    for competition in normalized_df['Competition'].unique():
        comp_mask = normalized_df['Competition'] == competition
        
        for col in metrics_to_normalize:
            if col in normalized_df.columns:
                col_values = normalized_df.loc[comp_mask, col]
                
                if col_values.nunique() > 1:
                    col_min = col_values.min()
                    col_max = col_values.max()
                    
                    if col_max > col_min:
                        # Invert for error-type metrics
                        if any(keyword in col.lower() for keyword in ['error', 'foul', 'card']):
                            normalized_df.loc[comp_mask, f'Normalized {col}'] = 1 - (col_values - col_min) / (col_max - col_min)
                        else:
                            normalized_df.loc[comp_mask, f'Normalized {col}'] = (col_values - col_min) / (col_max - col_min)
                    else:
                        normalized_df.loc[comp_mask, f'Normalized {col}'] = 0.5
                else:
                    normalized_df.loc[comp_mask, f'Normalized {col}'] = 0.5
    
    # Normalize percentage columns
    for col in percentage_cols:
        if col in normalized_df.columns:
            normalized_df[f'Normalized {col}'] = normalized_df[col] / 100
    
    return normalized_df

@st.cache_data
def calculate_similarity(team1_data, team2_data):
    """Calculate similarity score between two teams"""
    normalized_cols = [col for col in team1_data.index if col.startswith('Normalized')]
    
    squared_diff_sum = 0
    valid_metrics = 0
    
    for col in normalized_cols:
        if col in team1_data and col in team2_data:
            val1 = team1_data[col]
            val2 = team2_data[col]
            
            if pd.notna(val1) and pd.notna(val2):
                squared_diff_sum += (val1 - val2) ** 2
                valid_metrics += 1
    
    if valid_metrics == 0:
        return 0
    
    similarity = 1 - (np.sqrt(squared_diff_sum / valid_metrics) / np.sqrt(2))
    return max(0, min(1, similarity))

def create_team_metrics_chart(team_data, available_metrics, normalized=True):
    """Create bar charts for team metrics"""
    if normalized:
        metric_cols = [f'Normalized {m}' for m in available_metrics if f'Normalized {m}' in team_data.index]
        y_range = [0, 1]
        title_suffix = "Normalized (0-1 scale)"
    else:
        metric_cols = [m for m in available_metrics if m in team_data.index]
        y_range = None
        title_suffix = "Raw Values"
    
    if not metric_cols:
        return None
    
    values = team_data[metric_cols]
    labels = [m.replace('Normalized ', '') for m in metric_cols]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color='#1E3A8A',
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Team Metrics - {title_suffix}",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        yaxis=dict(range=y_range),
        template="plotly_white"
    )
    
    return fig

def create_comparison_chart(team1_data, team2_data, available_metrics, normalized=True):
    """Create comparison bar charts for two teams"""
    if normalized:
        metric_cols = [f'Normalized {m}' for m in available_metrics if 
                      f'Normalized {m}' in team1_data.index and f'Normalized {m}' in team2_data.index]
        y_range = [0, 1]
        title_suffix = "Normalized (0-1 scale)"
    else:
        metric_cols = [m for m in available_metrics if m in team1_data.index and m in team2_data.index]
        y_range = None
        title_suffix = "Raw Values"
    
    if not metric_cols:
        return None
    
    values1 = team1_data[metric_cols]
    values2 = team2_data[metric_cols]
    labels = [m.replace('Normalized ', '') for m in metric_cols]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values1,
        marker_color='#1E3A8A',
        name=team1_data['Squad'],
        text=[f"{v:.2f}" for v in values1],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values2,
        marker_color='#DC2626',
        name=team2_data['Squad'],
        text=[f"{v:.2f}" for v in values2],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Team Comparison - {title_suffix}",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode='group',
        height=500,
        yaxis=dict(range=y_range),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def main():
    """Main function to run the Streamlit application"""
    st.markdown('<p class="main-header">⚽ Football Team Playing Style Analyzer</p>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: right; color: #888;'>Last updated: {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'normalized_teams_df' not in st.session_state:
        st.session_state['normalized_teams_df'] = None
    
    # Load data
    with st.spinner("Loading data..."):
        normalized_teams_df = load_and_process_data()
        st.session_state['normalized_teams_df'] = normalized_teams_df
    
    if normalized_teams_df is not None:
        # Navigation
        st.sidebar.markdown("## Navigation")
        app_mode = st.sidebar.radio("Select Mode", ["Single Team Analysis", "Team Comparison"])
        
        # Get available data
        competitions = sorted(normalized_teams_df['Competition'].unique())
        seasons = sorted(normalized_teams_df['Season'].unique(), reverse=True)
        
        # Season filter
        selected_season = st.sidebar.selectbox("Select Season", seasons)
        season_mask = normalized_teams_df['Season'] == selected_season
        
        # Get available metrics dynamically
        all_metrics = [col for col in normalized_teams_df.columns if 
                      col not in ['Squad', 'Competition', 'Season'] and 
                      not col.startswith('Normalized')]
        
        if app_mode == "Single Team Analysis":
            st.markdown('<p class="sub-header">Single Team Analysis</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_competition = st.selectbox("Select Competition", competitions)
                comp_mask = normalized_teams_df['Competition'] == selected_competition
                filtered_df = normalized_teams_df[comp_mask & season_mask]
                teams = sorted(filtered_df['Squad'].unique())
            
            with col2:
                selected_team = st.selectbox("Select Team", teams)
            
            if teams and selected_team:
                team_mask = filtered_df['Squad'] == selected_team
                if not filtered_df[team_mask].empty:
                    team_data = filtered_df[team_mask].iloc[0]
                    
                    show_normalized = st.checkbox("Show Normalized Values", value=True)
                    
                    # Display available metrics
                    team_metrics = [col for col in all_metrics if col in team_data.index and pd.notna(team_data[col])]
                    
                    if team_metrics:
                        chart = create_team_metrics_chart(team_data, team_metrics, normalized=show_normalized)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Show metrics table
                        st.markdown("### Team Metrics")
                        metrics_data = []
                        for metric in team_metrics[:15]:  # Show top 15 metrics
                            if metric in team_data.index:
                                metrics_data.append({
                                    'Metric': metric,
                                    'Value': f"{team_data[metric]:.2f}"
                                })
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df, use_container_width=True)
                    else:
                        st.warning("No metrics available for this team.")
                else:
                    st.warning("No data available for the selected team.")
        
        elif app_mode == "Team Comparison":
            st.markdown('<p class="sub-header">Team Comparison</p>', unsafe_allow_html=True)
            
            # Team 1 selection
            st.markdown("### Select First Team")
            col1, col2 = st.columns(2)
            
            with col1:
                comp1 = st.selectbox("Competition (Team 1)", competitions, key="comp1")
                comp1_mask = normalized_teams_df['Competition'] == comp1
                filtered_df1 = normalized_teams_df[comp1_mask & season_mask]
                teams1 = sorted(filtered_df1['Squad'].unique())
            
            with col2:
                team1 = st.selectbox("Team 1", teams1, key="team1")
            
            # Team 2 selection
            st.markdown("### Select Second Team")
            col1, col2 = st.columns(2)
            
            with col1:
                comp2 = st.selectbox("Competition (Team 2)", competitions, key="comp2")
                comp2_mask = normalized_teams_df['Competition'] == comp2
                filtered_df2 = normalized_teams_df[comp2_mask & season_mask]
                teams2 = sorted(filtered_df2['Squad'].unique())
            
            with col2:
                team2 = st.selectbox("Team 2", teams2, key="team2")
            
            # Compare teams
            if teams1 and teams2 and team1 and team2:
                team1_mask = filtered_df1['Squad'] == team1
                team2_mask = filtered_df2['Squad'] == team2
                
                if not filtered_df1[team1_mask].empty and not filtered_df2[team2_mask].empty:
                    team1_data = filtered_df1[team1_mask].iloc[0]
                    team2_data = filtered_df2[team2_mask].iloc[0]
                    
                    # Calculate similarity
                    similarity = calculate_similarity(team1_data, team2_data)
                    
                    st.markdown(
                        f"""
                        <div class="similarity-circle">
                            <div class="similarity-value">{similarity*100:.1f}%</div>
                            <div class="similarity-label">Similarity</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    show_normalized = st.checkbox("Show Normalized Values", value=True)
                    
                    # Get common metrics
                    common_metrics = [col for col in all_metrics if 
                                    col in team1_data.index and col in team2_data.index and 
                                    pd.notna(team1_data[col]) and pd.notna(team2_data[col])]
                    
                    if common_metrics:
                        chart = create_comparison_chart(team1_data, team2_data, common_metrics, normalized=show_normalized)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Comparison table
                        st.markdown("### Key Metrics Comparison")
                        comparison_data = []
                        
                        for metric in common_metrics[:10]:  # Show top 10 metrics
                            val1 = team1_data[metric]
                            val2 = team2_data[metric]
                            
                            comparison_data.append({
                                'Metric': metric,
                                f'{team1}': f"{val1:.2f}",
                                f'{team2}': f"{val2:.2f}",
                                'Difference': f"{val1 - val2:.2f}"
                            })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                    else:
                        st.warning("No common metrics available for comparison.")
                else:
                    st.warning("Data missing for one or both selected teams.")
        
        # Instructions
        st.markdown("---")
        st.markdown('<p class="sub-header">How to Use This Dashboard</p>', unsafe_allow_html=True)
        st.markdown("""
        ### Features:
        - **Single Team Analysis**: View individual team performance metrics
        - **Team Comparison**: Compare two teams side-by-side with similarity scoring
        - **Normalized Values**: All metrics scaled 0-1 within each competition for fair comparison
        - **Dynamic Data**: Automatically adapts to available metrics in your dataset
        
        ### Data Requirements:
        The app loads real football data from your GitHub repository. Ensure your CSV contains:
        - Team/Squad column
        - Competition/League column  
        - Season column
        - Various performance metrics (goals, shots, passes, etc.)
        """)
    else:
        st.error("Error loading data. Please check if the CSV file is accessible and properly formatted.")

if __name__ == "__main__":
    main()
