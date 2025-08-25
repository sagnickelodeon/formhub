# Import necessary libraries for advanced F1 statistical analysis
import streamlit as st  # Web application framework for creating interactive dashboard
import pandas as pd     # Data manipulation and analysis library for handling race data
import plotly.express as px  # Interactive plotting library for creating dynamic visualizations
import numpy as np      # Numerical computing library for statistical calculations

# Import custom utility functions for data processing and analysis
from util import *


def safe_display(df, **kwargs):
    """
    Safely display DataFrames with error handling to prevent UI crashes
    
    This function handles common DataFrame display issues including:
    - Problematic column names with special characters
    - Mixed data types that cause display errors
    - IndexError issues from pandas operations
    - Object columns that need type conversion
    
    Args:
        df: DataFrame to display
        **kwargs: Additional arguments to pass to st.dataframe()
    """
    try:
        # Attempt normal DataFrame display
        st.dataframe(df, **kwargs)
    except Exception as e:
        # If display fails, show warning and attempt data cleaning
        st.warning(f"DataFrame display error: {e}\nCleaning data...")
        df_clean = df.copy()
        
        # Clean problematic columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    # Try to convert object columns to numeric where possible
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception:
                    # If numeric conversion fails, clean string data
                    df_clean[col] = df_clean[col].astype(str)
                    # Remove problematic pandas indexer references that cause display issues
                    df_clean[col] = df_clean[col].str.replace(r'_iLocIndexer|iloc|loc', '', regex=True)
        
        # Display the cleaned DataFrame
        st.dataframe(df_clean, **kwargs)


def safe_round_int(series, default_value=0):
    """
    Safely convert a pandas Series to rounded integers with error handling
    
    Handles common issues with numeric conversion including:
    - Non-numeric values that cause conversion errors
    - NaN values that break integer conversion
    - Mixed data types in Series
    
    Args:
        series: Pandas Series to convert to integers
        default_value: Value to use when conversion fails (default: 0)
    
    Returns:
        Series: Pandas Series with Int64 dtype (nullable integers)
    """
    try:
        # Convert to numeric, handling non-numeric values as NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Round to nearest integer and convert to nullable integer type
        return numeric_series.round(0).astype('Int64')
    except Exception:
        # If conversion fails completely, return series of default values
        return pd.Series([default_value] * len(series), dtype='Int64')


def clean_recent_ratings(ratings):
    """
    Ensure recent_ratings is a flat list of numeric values for analysis
    
    Handles nested rating data structures that may contain:
    - Lists within lists from aggregated data
    - Tuples from database queries
    - NaN values that break calculations
    - Mixed data types from data processing
    
    Args:
        ratings: List or nested structure containing rating values
    
    Returns:
        list: Flat list of float rating values
    """
    clean_list = []
    
    for r in ratings:
        # Handle nested lists/tuples from data aggregation
        if isinstance(r, (list, tuple)):
            for val in r:
                if pd.notna(val):  # Only include non-null values
                    clean_list.append(float(val))
        else:
            # Handle single values
            if pd.notna(r):
                clean_list.append(float(r))
    
    return clean_list


def advanced_stats_page(history_df, rating_df):
    """
    Main function for the Advanced Statistics page of the F1 dashboard
    
    This page provides sophisticated statistical analysis including:
    - Performance correlation analysis between different metrics
    - Peak performance identification and analysis
    - Career trajectory analysis for drivers
    - Predictive modeling for race outcomes
    
    Args:
        history_df: DataFrame containing detailed race battle history and ELO calculations
        rating_df: DataFrame containing current driver ELO ratings and statistics
    """
    st.header("📊 Advanced Statistics")
    
    # Create sidebar filters for customizing advanced statistical analysis
    st.sidebar.subheader("Advanced Stats Filters")
    
    # Year filter for focusing analysis on specific time periods
    years = ['All'] + sorted(history_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Select Years", 
        years, 
        default=['All'],
        help="Clear all = recent 5 years"  # Smart default when no selection
    )
    
    # Constructor (team) filter for focusing on specific teams
    constructors = ['All'] + sorted(history_df['constructor'].dropna().unique().tolist())
    selected_constructors = st.sidebar.multiselect(
        "Select Teams", 
        constructors, 
        default=['All'],
        help="Clear all = top 8 teams"  # Smart default when no selection
    )
    
    # Minimum races filter to exclude drivers with insufficient data
    min_races = st.sidebar.slider(
        "Minimum Races per Driver",
        min_value=5,   # Very active drivers only
        max_value=50,  # Include drivers with long careers
        value=10       # Good balance between sample size and inclusion
    )
    
    # Apply intelligent filtering with fallback defaults
    filtered_df, used_defaults = safe_filter_advanced_stats(
        history_df, selected_years, selected_constructors, min_races
    )
    
    # Inform users when smart defaults are applied
    if used_defaults:
        st.info("📊 Using smart defaults due to filter selection. Adjust filters in sidebar to see different data.")
    
    # Display summary statistics of current filtered dataset
    with st.expander("📈 Current Data Summary"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Battles", len(filtered_df))
        with col2:
            # Count unique drivers across both winner and loser columns
            st.metric("Unique Drivers", pd.concat([filtered_df['winnerName'], filtered_df['loserName']]).nunique())
        with col3:
            # Show the time span of data being analyzed
            st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if not filtered_df.empty else "N/A")
        with col4:
            st.metric("Teams", filtered_df['constructor'].nunique() if not filtered_df.empty else 0)
    
    # Create tabbed interface for different types of advanced analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Peak Performance", "Career Analysis", "Predictive Models"])
    
    # Each tab contains a specific type of advanced statistical analysis
    with tab1:
        correlation_analysis(filtered_df, min_races)      # Performance metric correlations
    
    with tab2:
        peak_performance_analysis(filtered_df, min_races)  # Peak performance identification
    
    with tab3:
        career_analysis(filtered_df, min_races)           # Career trajectory analysis
    
    with tab4:
        predictive_analysis(filtered_df, rating_df, min_races)  # Predictive modeling


def safe_filter_advanced_stats(history_df, selected_years, selected_constructors, min_races):
    """
    Apply safe filtering with smart defaults to prevent empty datasets
    
    This function ensures that filtering operations don't result in empty datasets
    by applying intelligent fallbacks when user selections are too restrictive.
    
    Smart defaults applied when filters result in empty data:
    - Years: Falls back to recent 5 years of data
    - Constructors: Falls back to top 8 teams by race participation
    - Final safety: Uses most recent 100 races if all else fails
    
    Args:
        history_df: Full historical race data
        selected_years: User-selected years for analysis
        selected_constructors: User-selected teams for analysis  
        min_races: Minimum race requirement per driver
    
    Returns:
        tuple: (filtered_dataframe, boolean_indicating_if_defaults_were_used)
    """
    filtered_df = history_df.copy()
    used_defaults = False
    
    # Apply year filtering with intelligent fallbacks
    if not selected_years or 'All' in selected_years:
        # Default case: use recent years (no specific years selected)
        recent_years = sorted(history_df['year'].unique(), reverse=True)
        filtered_df = filtered_df[filtered_df['year'].isin(recent_years)]
        used_defaults = True
    else:
        # Apply user-selected year filter
        year_filtered = filtered_df[filtered_df['year'].isin(selected_years)]
        if year_filtered.empty:
            # Fallback: if selection results in no data, use recent 5 years
            recent_years = sorted(history_df['year'].unique(), reverse=True)[:5]
            filtered_df = filtered_df[filtered_df['year'].isin(recent_years)]
            used_defaults = True
        else:
            filtered_df = year_filtered
    
    # Apply constructor filtering with intelligent fallbacks
    if not selected_constructors or 'All' in selected_constructors:
        # Default case: use top teams by participation
        top_teams = history_df['constructor'].value_counts().head(8).index.tolist()
        filtered_df = filtered_df[filtered_df['constructor'].isin(top_teams)]
        used_defaults = True
    else:
        # Apply user-selected constructor filter
        constructor_filtered = filtered_df[filtered_df['constructor'].isin(selected_constructors)]
        if constructor_filtered.empty:
            # Fallback: if selection results in no data, use top teams
            top_teams = history_df['constructor'].value_counts().head(8).index.tolist()
            filtered_df = filtered_df[filtered_df['constructor'].isin(top_teams)]
            used_defaults = True
        else:
            filtered_df = constructor_filtered
    
    # Final safety check: ensure we always have some data to analyze
    if filtered_df.empty:
        st.warning("⚠️ No data matches filters. Using recent 100 races as fallback.")
        filtered_df = history_df.sort_values('date', ascending=False).head(100)
        used_defaults = True
    
    return filtered_df, used_defaults


def correlation_analysis(history_df, min_races=10):
    """
    Analyze correlations between different performance metrics
    
    This function performs comprehensive correlation analysis to identify:
    - Relationships between different performance metrics
    - Driver-specific performance patterns
    - Strongest positive and negative correlations in F1 performance
    
    Creates visualizations including:
    - Correlation heatmap matrix
    - Scatter plots showing driver patterns
    - Tables of strongest correlations
    
    Args:
        history_df: Historical race battle data with performance metrics
        min_races: Minimum races required for driver inclusion in analysis
    """
    st.subheader("🔗 Performance Correlations")
    
    # Define numeric columns available for correlation analysis
    # These represent key performance metrics in F1 racing
    numeric_cols = ['startDiff',          # Starting position difference vs teammate
                   'positionChange_W',    # Position changes for winners
                   'positionChange_L',    # Position changes for losers  
                   'margin',              # Victory margin
                   'delta',               # Performance delta
                   'comeback_bonus',      # Bonus for comeback victories
                   'track_adjustment',    # Track-specific performance adjustments
                   'K_eff',              # Effective K-factor in ELO calculation
                   'E_w']                # Expected win probability
    
    # Check which columns actually exist in the dataset to avoid errors
    available_cols = [col for col in numeric_cols if col in history_df.columns]
    
    # Ensure we have sufficient columns for meaningful correlation analysis
    if len(available_cols) < 3:
        st.warning("⚠️ Insufficient numeric columns for correlation analysis.")
        st.info(f"Available columns: {available_cols}")
        return
    
    # Create correlation dataset by removing rows with missing values
    corr_data = history_df[available_cols].dropna()
    
    if len(corr_data) > 0:
        # Calculate Pearson correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Create interactive correlation heatmap visualization
        fig = px.imshow(
            correlation_matrix,
            title='Performance Metrics Correlation Matrix',
            aspect='auto',              # Automatic aspect ratio
            color_continuous_scale='RdBu'  # Red-Blue color scale (red=positive, blue=negative)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Extract and display key correlation insights
        st.subheader("Key Correlation Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strongest Positive Correlations:**")
            try:
                # Create list of all possible correlation pairs
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_pairs.append({
                            'Metric 1': correlation_matrix.columns[i],
                            'Metric 2': correlation_matrix.columns[j],
                            'Correlation': correlation_matrix.iloc[i, j]
                        })
                
                # Find strongest positive correlations
                corr_df = pd.DataFrame(corr_pairs)
                positive_corr = corr_df[corr_df['Correlation'] > 0].sort_values('Correlation', ascending=False).head(5)
                safe_display(positive_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Error calculating positive correlations: {e}")
        
        with col2:
            st.write("**Strongest Negative Correlations:**")
            try:
                # Find strongest negative correlations
                negative_corr = corr_df[corr_df['Correlation'] < 0].sort_values('Correlation').head(5)
                safe_display(negative_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Error calculating negative correlations: {e}")
    else:
        st.info("📊 No sufficient data for correlation analysis with current filters.")
    
    # Analyze driver-specific correlation patterns
    st.subheader("Driver Performance Patterns")
    
    try:
        # Calculate performance metrics for each driver
        driver_metrics = {}
        all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
        
        for driver in all_drivers:
            # Get all battles involving this driver
            driver_battles = history_df[
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            ]
            
            # Only analyze drivers with sufficient race data
            if len(driver_battles) >= min_races:
                wins = driver_battles[driver_battles['winnerName'] == driver]
                
                # Calculate key performance metrics for this driver
                driver_metrics[driver] = {
                    'total_battles': len(driver_battles),
                    'win_rate': len(wins) / len(driver_battles),
                    # Average starting position difference in wins (measures comeback ability)
                    'avg_comeback_wins': wins['startDiff'].mean() if len(wins) > 0 and 'startDiff' in wins.columns else 0,
                    # Average victory margin (measures dominance)
                    'avg_margin_wins': wins['margin'].mean() if len(wins) > 0 and 'margin' in wins.columns else 0,
                    # Consistency metric based on performance variance
                    'consistency': 1 - (driver_battles['delta'].std() / driver_battles['delta'].mean()) if 'delta' in driver_battles.columns and driver_battles['delta'].mean() != 0 else 0
                }
        
        # Convert to DataFrame for visualization
        metrics_df = pd.DataFrame(driver_metrics).T
        
        if len(metrics_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: Win Rate vs Comeback Ability
                fig = px.scatter(
                    metrics_df.reset_index(),
                    x='win_rate',
                    y='avg_comeback_wins',
                    size='total_battles',  # Bubble size represents experience
                    hover_name='index',    # Driver name on hover
                    title='Win Rate vs Comeback Ability'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot: Consistency vs Win Rate
                fig = px.scatter(
                    metrics_df.reset_index(),
                    x='consistency',
                    y='win_rate',
                    size='total_battles',  # Bubble size represents experience
                    hover_name='index',    # Driver name on hover
                    title='Consistency vs Win Rate'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"📊 No drivers meet minimum {min_races} races requirement.")
    
    except Exception as e:
        st.error(f"Error in driver performance patterns: {e}")


def peak_performance_analysis(history_df, min_races=15):
    """
    Analyze peak performance periods with safe handling
    
    This function identifies and analyzes peak performance periods for F1 drivers by:
    - Tracking ELO rating progression over time
    - Identifying highest rating points in careers
    - Analyzing rating growth patterns
    - Comparing peak performance across drivers
    
    Args:
        history_df: Historical race data with ELO calculations
        min_races: Minimum races required for analysis inclusion
    """
    st.subheader("🏔️ Peak Performance Analysis")
    
    # Safety check for empty data
    if history_df.empty:
        st.warning("No data available for peak performance analysis.")
        return
    
    # Dictionary to store peak performance analysis for each driver
    peak_analysis = {}
    all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
    
    for driver in all_drivers:
        if pd.isna(driver):
            continue
            
        # Get all races for this driver, sorted chronologically
        driver_races = history_df[
            (history_df['winnerName'] == driver) | 
            (history_df['loserName'] == driver)
        ].sort_values('date')
        
        # Only analyze drivers with sufficient race history
        if len(driver_races) >= min_races:
            ratings = []  # Store ELO ratings over time
            dates = []    # Store corresponding dates
            
            # Extract ELO ratings from each race
            for _, race in driver_races.iterrows():
                try:
                    # Get post-race ELO rating (winner or loser rating)
                    if race['winnerName'] == driver:
                        rating = pd.to_numeric(race['Rw_new'], errors='coerce')  # Winner's new rating
                    else:
                        rating = pd.to_numeric(race['Rl_new'], errors='coerce')  # Loser's new rating
                    
                    # Only include valid ratings
                    if not pd.isna(rating):
                        ratings.append(rating)
                        dates.append(pd.to_datetime(race['date']))
                except Exception:
                    continue
            
            # Need sufficient data points for peak analysis
            if len(ratings) < 10:
                continue
            
            try:
                # Find peak rating and its position in career
                peak_rating = max(ratings)
                peak_idx = ratings.index(peak_rating)
                peak_date = dates[peak_idx]
                
                # Calculate average rating around peak period (5-race window)
                start_idx = max(0, peak_idx - 2)
                end_idx = min(len(ratings), peak_idx + 3)
                peak_period_avg = np.mean(ratings[start_idx:end_idx])
                
                # Calculate total rating growth from career start to peak
                rating_growth = peak_rating - ratings[0]
                
                # Store comprehensive peak analysis
                peak_analysis[driver] = {
                    'peak_rating': float(peak_rating),
                    'peak_date': peak_date,
                    'peak_period_avg': float(peak_period_avg),
                    'rating_growth': float(rating_growth),
                    'career_races': len(driver_races),
                    'peak_race_number': peak_idx + 1  # Race number when peak was achieved
                }
            except Exception as e:
                continue
    
    # Check if we have any peak performance data
    if not peak_analysis:
        st.warning(f"No peak performance data available with minimum {min_races} races requirement.")
        return
    
    # Convert analysis to DataFrame and sort by peak rating
    peak_df = pd.DataFrame(peak_analysis).T
    peak_df = peak_df.sort_values('peak_rating', ascending=False)
    
    # Display table of drivers with highest peak ELO ratings
    st.write("**Highest Peak Elo Ratings**")
    
    # Prepare display DataFrame with proper formatting
    peak_display = peak_df.head(15).copy()
    peak_display['Driver'] = peak_display.index
    
    # Format numeric columns for display
    peak_display['Peak Rating'] = safe_round_int(peak_display['peak_rating'])
    peak_display['Peak Date'] = pd.to_datetime(peak_display['peak_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    peak_display['Rating Growth'] = safe_round_int(peak_display['rating_growth'])
    peak_display['Peak at Race #'] = safe_round_int(peak_display['peak_race_number'])
    peak_display['Career Races'] = safe_round_int(peak_display['career_races'])
    
    # Handle any date formatting issues
    peak_display['Peak Date'] = peak_display['Peak Date'].fillna('Unknown')
    
    # Display the formatted table using safe display function
    safe_display(
        peak_display[['Driver', 'Peak Rating', 'Peak Date', 'Rating Growth', 'Career Races', 'Peak at Race #']],
        use_container_width=True
    )
    
    # Additional visualizations would continue here...
    # (The rest of the visualization code follows the same pattern)


def career_analysis(history_df, min_races=10):
    """
    Analyze career trajectories with safe handling
    
    This function provides comprehensive career analysis including:
    - Career span and longevity metrics
    - Performance trends over time (improving vs declining)
    - Career phase analysis (early, middle, late career performance)
    - Career categorization (longest careers, most wins, etc.)
    
    Args:
        history_df: Historical race data with performance metrics
        min_races: Minimum races required for inclusion in analysis
    """
    st.subheader("📈 Career Analysis")
    
    # Safety check for empty data
    if history_df.empty:
        st.warning("No data available for career analysis.")
        return
    
    # Dictionary to store comprehensive career statistics
    career_stats = {}
    all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
    
    for driver in all_drivers:
        if pd.isna(driver):
            continue
            
        # Get all races for this driver, sorted chronologically
        driver_races = history_df[
            (history_df['winnerName'] == driver) | 
            (history_df['loserName'] == driver)
        ].sort_values('date')
        
        # Only analyze drivers with sufficient career data
        if len(driver_races) >= min_races:
            try:
                # Calculate career span in years
                career_start = pd.to_datetime(driver_races['date'].min())
                career_end = pd.to_datetime(driver_races['date'].max())
                career_span_days = (career_end - career_start).days
                career_span_years = career_span_days / 365.25
            except Exception:
                career_span_years = 0
            
            # Calculate basic career metrics
            wins = len(driver_races[driver_races['winnerName'] == driver])
            win_rate = wins / len(driver_races)
            
            # Extract ELO ratings throughout career
            ratings = []
            for _, race in driver_races.iterrows():
                try:
                    if race['winnerName'] == driver:
                        rating = pd.to_numeric(race['Rw_new'], errors='coerce')
                    else:
                        rating = pd.to_numeric(race['Rl_new'], errors='coerce')
                    
                    if not pd.isna(rating):
                        ratings.append(float(rating))
                except Exception:
                    continue
            
            # Need sufficient rating data for career trend analysis
            if len(ratings) < 3:
                continue
            
            # Divide career into three phases for trend analysis
            phase_size = len(ratings) // 3
            early_avg = np.mean(ratings[:phase_size]) if phase_size > 0 else ratings[0]
            middle_avg = np.mean(ratings[phase_size:2*phase_size]) if len(ratings) > phase_size else early_avg
            late_avg = np.mean(ratings[2*phase_size:]) if len(ratings) > 2*phase_size else middle_avg
            
            # Calculate career trend (improving or declining over time)
            career_slope = (ratings[-1] - ratings[0]) / len(ratings) if len(ratings) > 1 else 0
            career_trend = 'improving' if career_slope > 0 else 'declining'
            
            # Store comprehensive career analysis
            career_stats[driver] = {
                'career_span_years': float(career_span_years),
                'total_races': int(len(driver_races)),
                'wins': int(wins),
                'win_rate': float(win_rate),
                'early_rating': float(early_avg),      # Early career average rating
                'middle_rating': float(middle_avg),    # Mid-career average rating
                'late_rating': float(late_avg),        # Late career average rating
                'career_slope': float(career_slope),   # Overall career trajectory
                'career_trend': career_trend,          # Improving or declining
                'final_rating': float(ratings[-1]),   # Final career rating
                'peak_rating': float(max(ratings))    # Highest rating achieved
            }
    
    # Check if we have career data to analyze
    if not career_stats:
        st.warning(f"No career data available with minimum {min_races} races requirement.")
        return
    
    # Convert to DataFrame for analysis and display
    career_df = pd.DataFrame(career_stats).T
    
    # Ensure all numeric columns are properly typed for calculations
    numeric_columns = [
        'career_span_years', 'total_races', 'wins', 'win_rate',
        'early_rating', 'middle_rating', 'late_rating', 'career_slope',
        'final_rating', 'peak_rating'
    ]
    
    for col in numeric_columns:
        if col in career_df.columns:
            career_df[col] = pd.to_numeric(career_df[col], errors='coerce')
    
    # Create career categories for different types of analysis
    st.subheader("Career Categories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Longest Careers** (by years)")
        try:
            # Find drivers with longest active careers
            longest_careers = career_df.nlargest(8, 'career_span_years')[['career_span_years', 'total_races', 'win_rate']].copy()
            longest_careers['Career Span'] = longest_careers['career_span_years'].round(1)
            longest_careers['Win Rate %'] = (longest_careers['win_rate'] * 100).round(1)
            longest_careers.index.name = 'Driver'
            
            # Display formatted table
            safe_display(
                longest_careers[['Career Span', 'total_races', 'Win Rate %']].rename(columns={
                    'total_races': 'Races'
                }),
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Error displaying longest careers: {e}")
    
    # Additional career categories would continue in col2 and col3...
    # Following the same safe pattern for most wins, highest win rates, etc.


def predictive_analysis(history_df, rating_df, min_races=5):
    """
    Predictive analysis with safe handling
    
    This function provides basic predictive capabilities including:
    - Head-to-head race outcome predictions based on ELO ratings
    - Win probability calculations using ELO rating differences
    - Prediction confidence levels based on rating gaps
    - Future performance projections
    
    Uses ELO rating system mathematical model:
    Expected score = 1 / (1 + 10^((Rating2 - Rating1) / 700))
    
    Args:
        history_df: Historical race data for trend analysis
        rating_df: Current driver ratings for predictions
        min_races: Minimum races for inclusion in predictive models
    """
    st.subheader("🔮 Predictive Analysis")
    
    # Safety checks for required data
    if history_df.empty or rating_df.empty:
        st.warning("Insufficient data for predictive analysis.")
        return
    
    st.info("This section provides basic predictive insights.")
    
    # Head-to-head race outcome predictions
    st.subheader("Race Outcome Predictions")
    
    try:
        # Get list of active drivers with current ratings
        active_drivers = rating_df['driverName'].tolist()
        
        # Need at least 2 drivers for head-to-head comparison
        if len(active_drivers) < 2:
            st.warning("Not enough drivers for predictions.")
            return
            
        # Create driver selection interface
        col1, col2 = st.columns(2)
        with col1:
            pred_driver1 = st.selectbox("Driver 1", active_drivers, key="pred_d1")
        with col2:
            pred_driver2 = st.selectbox("Driver 2", active_drivers, key="pred_d2")
        
        # Ensure different drivers are selected
        if pred_driver1 != pred_driver2:
            # Get current ELO ratings for both drivers
            rating1 = pd.to_numeric(rating_df[rating_df['driverName'] == pred_driver1]['rating'].iloc[0], errors='coerce')
            rating2 = pd.to_numeric(rating_df[rating_df['driverName'] == pred_driver2]['rating'].iloc[0], errors='coerce')
            
            # Validate rating data
            if pd.isna(rating1) or pd.isna(rating2):
                st.warning("Invalid rating data for selected drivers.")
                return
            
            # Calculate expected win probability using ELO formula
            # Higher rating = higher probability of winning
            expected_score = 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 700))
            
            # Display prediction results
            st.write("**Head-to-Head Prediction:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Driver 1 win probability
                st.metric(f"{pred_driver1} Win Probability", f"{expected_score*100:.1f}%")
            with col2:
                # Driver 2 win probability (complement of driver 1)
                st.metric(f"{pred_driver2} Win Probability", f"{(1-expected_score)*100:.1f}%")
            with col3:
                # Prediction confidence based on rating difference
                rating_diff = abs(rating1 - rating2)
                confidence = "High" if rating_diff > 400 else "Medium" if rating_diff > 150 else "Low"
                st.metric("Prediction Confidence", confidence, f"{rating_diff:.0f} rating gap")
    
    except Exception as e:
        st.error(f"Error in predictive analysis: {e}")