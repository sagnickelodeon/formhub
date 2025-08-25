# Import required libraries for data processing, visualization, and statistical analysis
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np

# Import utility functions from local util module
from util import *


def performance_analytics_page(history_df, rating_df, result_df):
    """
    Main function to render the F1 performance analytics page.
    Provides comprehensive performance analysis through multiple analytical lenses:
    - Comeback analysis (drivers who excel at recovering from poor grid positions)
    - Race craft analysis (qualifying vs race day performance comparison)
    - Form guide (current performance trends and recent results)
    - Consistency analysis (performance volatility and reliability metrics)
    
    Args:
        history_df: DataFrame containing historical teammate battle data
        rating_df: DataFrame containing rating information (currently unused)
        result_df: DataFrame containing race results and final positions
    """
    # Display page header with trending chart emoji
    st.header("📈 Performance Analytics")
    
    # **STEP 1: Add sidebar filters for safe filtering**
    # Create sidebar section for performance-specific filters
    st.sidebar.subheader("Performance Analytics Filters")
    
    # Year filter - multi-select with reverse chronological order (newest first)
    years = ['All'] + sorted(history_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Select Years", 
        years, 
        default=['All'],  # Default to all years
        help="Clear all = recent 5 years"  # Tooltip explaining fallback behavior
    )
    
    # Constructor/team filter - multi-select with alphabetical sorting
    constructors = ['All'] + sorted(history_df['constructor'].dropna().unique().tolist())
    selected_constructors = st.sidebar.multiselect(
        "Select Teams", 
        constructors, 
        default=['All'],  # Default to all teams
        help="Clear all = all teams"  # Tooltip explaining fallback behavior
    )
    
    # Minimum races filter - slider to ensure statistical significance for performance metrics
    min_races = st.sidebar.slider(
        "Minimum Races per Driver",
        min_value=3,   # Very low minimum for performance analysis
        max_value=30,  # High enough for veteran drivers
        value=5        # Default requires 5 races for meaningful performance assessment
    )
    
    # **STEP 2: Apply safe filtering**
    # Apply filters using safe filtering function with performance-specific fallback handling
    filtered_df, used_defaults = safe_filter_performance_analytics(
        history_df, selected_years, selected_constructors
    )
    
    # Display current filtered data status in expandable section
    with st.expander("📈 Current Data Summary"):
        # Create four columns for key performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total battles available for analysis
        with col1:
            st.metric("Total Battles", len(filtered_df))
        
        # Column 2: Number of unique drivers in performance analysis
        with col2:
            st.metric("Unique Drivers", pd.concat([filtered_df['winnerName'], filtered_df['loserName']]).nunique())
        
        # Column 3: Time span of performance data
        with col3:
            st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if not filtered_df.empty else "N/A")
        
        # Column 4: Number of different teams/constructors represented
        with col4:
            st.metric("Teams", filtered_df['constructor'].nunique() if not filtered_df.empty else 0)
    
    # **STEP 3: Tabs with filtered data**
    # Create four analytical tabs, each focusing on different performance aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Comeback Analysis", "Race Craft", "Form Guide", "Consistency"])
    
    # Tab 1: Analyze drivers who excel at recovering from poor starting positions
    with tab1:
        comeback_analysis(filtered_df, min_races)
    
    # Tab 2: Compare qualifying performance vs race day performance
    with tab2:
        race_craft_analysis(filtered_df, min_races)
    
    # Tab 3: Current form and recent performance trends
    with tab3:
        form_guide_analysis(filtered_df, result_df, min_races)
    
    # Tab 4: Performance consistency and volatility analysis
    with tab4:
        consistency_analysis(filtered_df, min_races)


def safe_filter_performance_analytics(history_df, selected_years, selected_constructors):
    """
    Apply safe filtering with smart defaults specifically for performance analytics.
    Uses larger fallback datasets since performance analysis needs more statistical data.
    
    Args:
        history_df: DataFrame with historical battle data
        selected_years: List of years to include ('All' for no filter)
        selected_constructors: List of constructors/teams to include ('All' for no filter)
    
    Returns:
        tuple: (filtered_history_df, used_defaults_boolean)
    """
    # Start with copy of original data to avoid modifying source
    filtered_df = history_df.copy()
    used_defaults = False  # Track if we had to use fallback logic
    
    try:
        # Apply year filtering with smart defaults
        # Check if no years selected or 'All' option chosen
        if not selected_years or 'All' in selected_years:
            # Default: use all available recent years for comprehensive performance analysis
            recent_years = sorted(history_df['year'].dropna().unique(), reverse=True)
            filtered_df = filtered_df[filtered_df['year'].isin(recent_years)]
            used_defaults = True
        else:
            # Filter by user-selected years
            year_filtered = filtered_df[filtered_df['year'].isin(selected_years)]
            # Check if year filtering resulted in empty dataset
            if year_filtered.empty:
                # Fallback to recent 5 years if selection too restrictive
                recent_years = sorted(history_df['year'].dropna().unique(), reverse=True)[:5]
                filtered_df = filtered_df[filtered_df['year'].isin(recent_years)]
                used_defaults = True
            else:
                # Use the year-filtered data
                filtered_df = year_filtered
        
        # Apply constructor/team filtering
        # Check if no constructors selected or 'All' option chosen
        if not selected_constructors or 'All' in selected_constructors:
            # Keep all constructors (no filtering needed)
            pass
        else:
            # Filter by user-selected constructors
            constructor_filtered = filtered_df[filtered_df['constructor'].isin(selected_constructors)]
            # Check if constructor filtering resulted in empty dataset
            if constructor_filtered.empty:
                st.warning(f"No data found for selected teams. Showing all teams.")
            else:
                # Apply constructor filter
                filtered_df = constructor_filtered
        
        # Final safety check - ensure we have adequate data for performance analysis
        if filtered_df.empty:
            st.warning("⚠️ Filters resulted in no data. Using recent 200 battles as fallback.")
            # Use most recent 200 battles as absolute fallback (larger than other pages)
            filtered_df = history_df.sort_values('date', ascending=False).head(200)
            used_defaults = True
        
        # Ensure minimum data for meaningful performance analysis
        # Performance analytics requires more data points than basic comparisons
        if len(filtered_df) < 50:
            st.info("📊 Limited data with current filters. Expanding to show more results...")
            # Get more recent data for better statistical significance
            recent_data = history_df.sort_values('date', ascending=False).head(500)
            if len(recent_data) > len(filtered_df):
                filtered_df = recent_data
                used_defaults = True
    
    except Exception as e:
        # Handle any unexpected errors in filtering process
        st.error(f"Error in filtering: {e}")
        # Use last 300 battles as emergency fallback (larger dataset for performance analysis)
        filtered_df = history_df.tail(300)
        used_defaults = True
    
    return filtered_df, used_defaults


# **KEEP ALL ANALYSIS FUNCTIONS WITH SAFE HANDLING**


def comeback_analysis(history_df, min_races=5):
    """
    Analyze drivers who excel at comeback performances from poor starting positions.
    Identifies "comeback kings" who consistently gain positions during races.
    Focuses on drivers who win despite starting from deficit positions on the grid.
    
    Args:
        history_df: DataFrame containing battle history with grid and race position data
        min_races: Minimum number of comeback wins required for inclusion (default: 5)
    """
    # Display section header with checkered flag emoji
    st.subheader("🏁 Comeback Kings")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for comeback analysis")
        return
    
    # Interactive filters with validation for comeback analysis
    # Minimum grid position deficit to qualify as a comeback
    min_deficit = st.slider("Minimum Grid Deficit", 1, 10, 3)
    # Minimum number of comeback victories for statistical significance
    min_comebacks = st.slider("Minimum Comebacks", 1, 20, 5)
    
    try:
        # Check that required columns exist for comeback analysis
        required_cols = ['startDiff', 'S_w', 'winnerName']
        missing_cols = [col for col in required_cols if col not in history_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        
        # Calculate comeback statistics with safe filtering
        # Filter for battles where winner started from a deficit (startDiff >= min_deficit)
        # and actually won the race (S_w == 1.0, meaning superior race result)
        comebacks = history_df[
            (pd.to_numeric(history_df['startDiff'], errors='coerce') >= min_deficit) & 
            (pd.to_numeric(history_df['S_w'], errors='coerce') == 1.0)
        ].copy()
        
        # Handle case where no comebacks found with current criteria
        if comebacks.empty:
            st.info(f"📊 No comebacks found with minimum deficit of {min_deficit} positions. Try reducing the minimum deficit.")
            return
        
        # Safe aggregation of comeback statistics per driver
        comeback_stats = comebacks.groupby('winnerName').agg({
            'startDiff': ['count', 'mean', 'max'],      # Count, average deficit, biggest deficit
            'comeback_bonus': 'mean',                   # Average comeback bonus multiplier
            'delta': 'mean',                           # Average Elo rating gain from comebacks
            'margin': 'mean'                           # Average victory margin in comeback wins
        }).round(2)
        
        # Flatten multi-level column names for easier access
        comeback_stats.columns = ['Total_Comebacks', 'Avg_Deficit', 'Max_Deficit', 'Avg_Bonus', 'Avg_Rating_Gain', 'Avg_Margin']
        # Filter drivers by minimum comeback requirement and sort by total comebacks
        comeback_stats = comeback_stats[comeback_stats['Total_Comebacks'] >= min_comebacks].sort_values('Total_Comebacks', ascending=False)
        
        # Display key comeback metrics with safe handling
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Top comeback specialist
        with col1:
            if len(comeback_stats) > 0:
                # Get driver with most comeback victories
                top_comeback_king = comeback_stats.index[0]
                top_comebacks = comeback_stats['Total_Comebacks'].iloc[0]
                st.metric("Comeback King", top_comeback_king, f"{int(top_comebacks)} comebacks")
            else:
                # Show requirement if no drivers qualify
                st.metric("Comeback King", "N/A", f"Min {min_comebacks} required")
        
        # Column 2: Biggest single comeback performance
        with col2:
            if not comebacks.empty:
                # Find the single biggest comeback (highest startDiff)
                biggest_comeback_idx = comebacks['startDiff'].idxmax()
                biggest_comeback = comebacks.loc[biggest_comeback_idx]
                st.metric(
                    "Biggest Comeback", 
                    f"{biggest_comeback['winnerName']}", 
                    f"{int(pd.to_numeric(biggest_comeback['startDiff'], errors='coerce'))} positions ({biggest_comeback['year']} {biggest_comeback['raceName']})"
                )
            else:
                st.metric("Biggest Comeback", "N/A", "No data")
        
        # Column 3: Average comeback bonus (Elo rating system bonus for difficult wins)
        with col3:
            if 'comeback_bonus' in comebacks.columns:
                # Calculate average bonus multiplier for comeback wins
                avg_comeback_bonus = pd.to_numeric(comebacks['comeback_bonus'], errors='coerce').mean()
                st.metric("Avg Comeback Bonus", f"{avg_comeback_bonus:.2f}" if pd.notna(avg_comeback_bonus) else "N/A", "Elo multiplier")
            else:
                st.metric("Avg Comeback Bonus", "N/A", "No data")
        
        # Display comeback specialists rankings table
        if len(comeback_stats) > 0:
            st.subheader("Comeback Specialists Rankings")
            display_df = comeback_stats.copy()
            display_df.index.name = 'Driver'  # Set index name for display
            safe_display(display_df, use_container_width=True)
            
            # Create visualization with comprehensive error handling
            try:
                # Ensure positive rating gains for visualization (clip negative values)
                comeback_stats['Avg_Rating_Gain'] = comeback_stats['Avg_Rating_Gain'].clip(lower=0)

                # Create scatter plot showing relationship between deficit size and comeback frequency
                fig = px.scatter(
                    comeback_stats.reset_index(),
                    x='Avg_Deficit',           # Average grid position deficit
                    y='Total_Comebacks',      # Number of comeback victories
                    size='Avg_Rating_Gain',   # Size bubbles by rating gain (reward for difficult wins)
                    hover_name='winnerName',   # Show driver name on hover
                    title='Comeback Analysis: Average Deficit vs Total Comebacks',
                    labels={'Avg_Deficit': 'Average Grid Deficit', 'Total_Comebacks': 'Total Comebacks'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create visualization: {viz_error}")
        
        # Show recent notable comebacks with safe handling
        if not comebacks.empty:
            # Define threshold for "notable" comebacks (minimum 8 position deficit)
            NOTABLE_COMEBACK_MIN_LIMIT = 8
            # Get most recent notable comebacks
            recent_comebacks = comebacks[comebacks["startDiff"] >= NOTABLE_COMEBACK_MIN_LIMIT].sort_values('date', ascending=False).head(10)
            st.subheader("Recent Notable Comebacks")
            
            # Prepare display-friendly format
            recent_display = recent_comebacks.copy()
            # Safe date formatting
            if 'date' in recent_display.columns:
                recent_display['Date'] = pd.to_datetime(recent_display['date']).dt.strftime('%Y-%m-%d')
            # Safe numeric conversion for deficit
            if 'startDiff' in recent_display.columns:
                recent_display['Deficit'] = pd.to_numeric(recent_display['startDiff'], errors='coerce').astype('Int64')
            # Safe numeric conversion for rating gain
            if 'delta' in recent_display.columns:
                recent_display['Rating Gain'] = pd.to_numeric(recent_display['delta'], errors='coerce').round(1)
            
            # Select and display available columns
            display_columns = ['Date', 'raceName', 'winnerName', 'Deficit', 'Rating Gain']
            available_columns = [col for col in display_columns if col in recent_display.columns]
            
            safe_display(
                recent_display[available_columns].rename(columns={
                    'raceName': 'Race',
                    'winnerName': 'Driver'
                }),
                use_container_width=True
            )
    
    except Exception as e:
        # Handle any errors in comeback analysis
        st.error(f"Error in comeback analysis: {e}")


def race_craft_analysis(history_df, min_races=5):
    """
    Analyze race day performance vs qualifying performance to identify different driver types.
    Distinguishes between "Race Day Heroes" (gain positions during races) and 
    "Qualifying Specialists" (start well, maintain positions).
    
    Args:
        history_df: DataFrame containing battle history with grid and race position data
        min_races: Minimum number of race wins required for inclusion (default: 5)
    """
    # Display section header with race car emoji
    st.subheader("🏎️ Race Craft Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for race craft analysis")
        return
    
    try:
        # Check for required columns and warn about limitations
        required_cols = ['positionChange_W', 'startDiff', 'winnerName']
        missing_cols = [col for col in required_cols if col not in history_df.columns]
        if missing_cols:
            st.warning(f"Some race craft data may be limited due to missing columns: {missing_cols}")
        
        # Calculate race day performance metrics with safe aggregation
        # Build aggregation dictionary based on available columns
        agg_dict = {}
        if 'positionChange_W' in history_df.columns:
            agg_dict['positionChange_W'] = 'mean'  # Average positions gained/lost during race
        if 'startDiff' in history_df.columns:
            agg_dict['startDiff'] = 'mean'          # Average starting position disadvantage
        if 'S_w' in history_df.columns:
            agg_dict['S_w'] = 'count'               # Total number of wins (race craft opportunities)
        if 'delta' in history_df.columns:
            agg_dict['delta'] = 'mean'              # Average Elo rating change per win
        
        # Exit if no suitable columns found
        if not agg_dict:
            st.error("No suitable columns found for race craft analysis")
            return
        
        # Aggregate statistics by winner (only analyze winning performances)
        race_craft_stats = history_df.groupby('winnerName').agg(agg_dict).round(2)
        
        # Safe column renaming with mapping dictionary
        column_mapping = {
            'positionChange_W': 'Avg_Pos_Change',   # Average positions gained during race
            'startDiff': 'Avg_Grid_Deficit',        # Average grid position deficit overcome
            'S_w': 'Total_Wins',                    # Total race victories
            'delta': 'Avg_Rating_Gain'              # Average Elo rating gain per win
        }
        
        # Apply column renaming safely
        new_columns = [column_mapping.get(col, col) for col in race_craft_stats.columns]
        race_craft_stats.columns = new_columns
        
        # Filter by minimum wins requirement for statistical significance
        if 'Total_Wins' in race_craft_stats.columns:
            race_craft_stats = race_craft_stats[race_craft_stats['Total_Wins'] >= min_races]
        
        # Handle case where no drivers meet minimum requirement
        if race_craft_stats.empty:
            st.info(f"📊 No drivers found with minimum {min_races} wins. Try reducing the minimum requirement.")
            return
        
        # Categorize drivers based on race craft characteristics with safe checks
        if 'Avg_Pos_Change' in race_craft_stats.columns:
            # Race Day Heroes: consistently gain positions during races (Avg_Pos_Change > 2)
            race_craft_stats['Race_Day_Hero'] = race_craft_stats['Avg_Pos_Change'] > 2
        if 'Avg_Pos_Change' in race_craft_stats.columns and 'Avg_Grid_Deficit' in race_craft_stats.columns:
            # Qualifying Specialists: start ahead and lose few positions (negative pos change, negative deficit)
            race_craft_stats['Quali_Specialist'] = (race_craft_stats['Avg_Pos_Change'] < 0) & (race_craft_stats['Avg_Grid_Deficit'] < 0)
        
        # Display driver categories in two-column layout
        col1, col2 = st.columns(2)
        
        # Column 1: Race Day Heroes (drivers who consistently gain positions)
        with col1:
            st.write("**Race Day Heroes** (Gain positions consistently)")
            if 'Race_Day_Hero' in race_craft_stats.columns and 'Avg_Pos_Change' in race_craft_stats.columns:
                # Filter and sort race day heroes by average position gain
                race_heroes = race_craft_stats[race_craft_stats['Race_Day_Hero']].sort_values('Avg_Pos_Change', ascending=False).head(10)
                if len(race_heroes) > 0:
                    # Display relevant columns for race day performance
                    display_cols = ['Avg_Pos_Change', 'Total_Wins']
                    available_cols = [col for col in display_cols if col in race_heroes.columns]
                    safe_display(race_heroes[available_cols], use_container_width=True)
                else:
                    st.info("No race day heroes found with current criteria")
            else:
                st.info("Position change data not available")
        
        # Column 2: Qualifying Specialists (drivers who start well and maintain advantage)
        with col2:
            st.write("**Qualifying Specialists** (Start ahead, stay ahead)")
            if 'Quali_Specialist' in race_craft_stats.columns and 'Avg_Grid_Deficit' in race_craft_stats.columns:
                # Filter and sort qualifying specialists by grid advantage
                quali_specialists = race_craft_stats[race_craft_stats['Quali_Specialist']].sort_values('Avg_Grid_Deficit').head(10)
                if len(quali_specialists) > 0:
                    # Display relevant columns for qualifying performance
                    display_cols = ['Avg_Grid_Deficit', 'Avg_Pos_Change', 'Total_Wins']
                    available_cols = [col for col in display_cols if col in quali_specialists.columns]
                    safe_display(quali_specialists[available_cols], use_container_width=True)
                else:
                    st.info("No qualifying specialists found with current criteria")
            else:
                st.info("Grid position data not available")
        
        # Create scatter plot visualization with safe handling
        if len(race_craft_stats) > 0 and 'Avg_Grid_Deficit' in race_craft_stats.columns and 'Avg_Pos_Change' in race_craft_stats.columns:
            try:
                # Create scatter plot showing race craft characteristics
                fig = px.scatter(
                    race_craft_stats.reset_index(),
                    x='Avg_Grid_Deficit',      # Starting disadvantage (negative = start ahead)
                    y='Avg_Pos_Change',       # Race day position changes (positive = gain positions)
                    size='Total_Wins' if 'Total_Wins' in race_craft_stats.columns else None,  # Bubble size by wins
                    hover_name='winnerName',   # Driver name on hover
                    title='Race Craft Analysis: Grid Position vs Position Changes',
                    labels={
                        'Avg_Grid_Deficit': 'Average Grid Deficit (negative = start ahead)',
                        'Avg_Pos_Change': 'Average Position Change (positive = gain positions)'
                    }
                )
                # Add reference lines for interpretation
                fig.add_hline(y=0, line_dash="dash", annotation_text="No position change")  # Neutral race performance
                fig.add_vline(x=0, line_dash="dash", annotation_text="Even grid start")    # Neutral grid position
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create race craft visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in race craft analysis
        st.error(f"Error in race craft analysis: {e}")


def form_guide_analysis(history_df, result_df, min_races=3):
    """
    Analyze current form and recent performance trends of drivers.
    Provides a form guide showing who's performing well recently and trending up/down.
    Combines race wins with final positions to give comprehensive recent form picture.
    
    Args:
        history_df: DataFrame containing battle history
        result_df: DataFrame containing race results with final positions
        min_races: Minimum number of recent races required for form assessment (default: 3)
    """
    # Display section header with bar chart emoji
    st.subheader("📊 Current Form Guide")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for form guide analysis")
        return
    
    # Form period selection with validation (how many recent races to analyze)
    form_races = st.slider("Form Guide Period (races)", 3, 25, 5)

    # Prepare result data for merging - focus on race winners (finalPosition == 1)
    result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
    result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

    # Merge history with race results to get complete picture
    history_df = pd.merge(history_df, result_filtered_df, on=["year", "round", "winnerId"])
    all_history_df = history_df.copy(deep=True)  # Keep all battles for context
    history_df = history_df[history_df["finalPosition"]==1]  # Focus on actual race wins
    
    try:
        # Calculate recent form with safe date handling
        # Define "recent" as last 9 months of racing activity
        max_date = pd.to_datetime(history_df['date']).max()
        recent_cutoff = max_date - pd.Timedelta(days=270)  # Last 9 months
        recent_history = history_df[pd.to_datetime(history_df['date']) >= recent_cutoff]
        recent_all_history = all_history_df[pd.to_datetime(all_history_df['date']) >= recent_cutoff]
        
        # Fallback if no recent data found
        if recent_history.empty:
            st.warning("No recent data found. Using all available data.")
            recent_history = history_df
        
        # Calculate form statistics for each driver
        form_stats = {}
        all_drivers = pd.concat([recent_history['winnerName'], recent_history['loserName']]).dropna().unique()
        
        # Process each driver's recent form
        for driver in all_drivers:
            if pd.isna(driver):
                continue

            # Get driver's recent ALL battles (for rating trend calculation)
            driver_all_races = recent_all_history[
                (recent_all_history['winnerName'] == driver) | 
                (recent_all_history['loserName'] == driver)
            ].sort_values('date', ascending=False).head(form_races)
            
            # Only analyze drivers with sufficient recent activity
            if len(driver_all_races) >= min_races:
                # Calculate win statistics (only counting actual race wins)
                wins = len(driver_all_races[(driver_all_races['winnerName'] == driver) & (driver_all_races["finalPosition"]==1)])
                total_races = len(driver_all_races)
                win_rate = wins / total_races
                
                # Calculate rating trend with safe handling
                ratings = []
                for _, race in driver_all_races.iterrows():
                    try:
                        # Get appropriate rating based on whether driver won or lost the battle
                        if race['winnerName'] == driver:
                            rating = pd.to_numeric(race['Rw_new'], errors='coerce')
                        else:
                            rating = pd.to_numeric(race['Rl_new'], errors='coerce')
                        
                        # Only add valid ratings to trend analysis
                        if pd.notna(rating):
                            ratings.append(float(rating))
                    except Exception:
                        continue
                
                # Determine rating trend and change
                if len(ratings) > 1:
                    # Compare most recent rating to oldest rating in period
                    rating_trend = 'improving' if ratings[-1] > ratings[0] else 'declining'
                    rating_change = ratings[-1] - ratings[0]
                    current_rating = ratings[-1]
                else:
                    # Insufficient data for trend analysis
                    rating_trend = 'stable'
                    rating_change = 0
                    current_rating = ratings[0] if ratings else 1500  # Default Elo rating
                
                # Store driver's form statistics
                form_stats[driver] = {
                    'recent_races': total_races,
                    'recent_wins': wins,
                    'form_percentage': win_rate * 100,      # Convert to percentage
                    'rating_trend': rating_trend,
                    'rating_change': rating_change,
                    'current_rating': current_rating
                }
        
        # Handle case where no drivers have sufficient recent activity
        if not form_stats:
            st.info(f"📊 No drivers found with minimum {min_races} recent races.")
            return
        
        # Convert to DataFrame and sort by current form (win percentage)
        form_df = pd.DataFrame(form_stats).T
        form_df = form_df.sort_values('form_percentage', ascending=False)
        
        # Display form guide table with driver performance trends
        st.write(f"**Form Guide - Last {form_races} Races**")
        
        # Prepare display-friendly format
        form_display = form_df.copy()
        form_display['Driver'] = form_display.index
        form_display['Win Rate %'] = pd.to_numeric(form_display['form_percentage'], errors='coerce').round(1)
        form_display['Rating Change'] = pd.to_numeric(form_display['rating_change'], errors='coerce').round(1)
        form_display['Current Rating'] = pd.to_numeric(form_display['current_rating'], errors='coerce').round(0).astype('Int64')
        # Add visual trend indicators using emojis
        form_display['Trend'] = form_display['rating_trend'].apply(lambda x: '📈' if x == 'improving' else '📉' if x == 'declining' else '➡️')
        
        # Display form guide table
        safe_display(
            form_display[['Driver', 'recent_wins', 'recent_races', 'Win Rate %', 'Current Rating', 'Rating Change', 'Trend']].rename(columns={
                'recent_wins': 'Wins',
                'recent_races': 'Races'
            }),
            use_container_width=True
        )
        
        # Create form visualization with error handling
        if len(form_df) > 0:
            try:
                # Show top 12 drivers by form for readability
                top_form = form_df.head(12)
                fig = px.bar(
                    top_form.reset_index(),
                    x='index',                  # Driver names on x-axis
                    y='form_percentage',        # Win rate percentage on y-axis
                    color='rating_change',      # Color bars by rating trend (red/green scale)
                    title=f'Current Form - Win Rate % (Last {form_races} Races)',
                    labels={'index': 'Driver', 'form_percentage': 'Win Rate %'},
                    color_continuous_scale='RdYlGn'  # Red (declining) to Green (improving) scale
                )
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create form visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in form guide analysis
        st.error(f"Error in form guide analysis: {e}")


def consistency_analysis(history_df, min_races=10):
    """
    Analyze driver consistency through performance volatility metrics.
    Identifies most consistent drivers (low volatility) vs most volatile drivers (high volatility).
    Uses rating changes and victory margins to measure performance consistency.
    
    Args:
        history_df: DataFrame containing battle history with rating changes
        min_races: Minimum number of battles required for consistency measurement (default: 10)
    """
    # Display section header with ruler emoji
    st.subheader("📏 Consistency Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for consistency analysis")
        return
    
    try:
        # Calculate consistency metrics for each driver
        consistency_stats = {}
        all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
        
        # Process each driver's consistency metrics
        for driver in all_drivers:
            if pd.isna(driver):
                continue
                
            # Get all battles involving this driver
            driver_battles = history_df[
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            ]
            
            # Only analyze drivers with sufficient battle history
            if len(driver_battles) >= min_races:
                # Collect rating changes and victory margins for volatility analysis
                rating_changes = []  # Elo rating changes per battle
                margins = []         # Victory/defeat margins per battle
                
                # Process each battle to extract performance metrics
                for _, race in driver_battles.iterrows():
                    try:
                        # Determine if driver won or lost and get appropriate metrics
                        if race['winnerName'] == driver:
                            # Driver won: get positive rating change and margin
                            delta = pd.to_numeric(race['delta'], errors='coerce')
                            margin = pd.to_numeric(race['margin'], errors='coerce')
                            if pd.notna(delta):
                                rating_changes.append(float(delta))
                            if pd.notna(margin):
                                margins.append(float(margin))
                        else:
                            # Driver lost: get negative rating change and margin
                            delta = pd.to_numeric(race['delta'], errors='coerce')
                            margin = pd.to_numeric(race['margin'], errors='coerce')
                            if pd.notna(delta):
                                rating_changes.append(-float(delta))  # Negative for losses
                            if pd.notna(margin):
                                margins.append(-float(margin))        # Negative for defeats
                    except Exception:
                        continue
                
                # Need minimum data points for meaningful consistency analysis
                if len(rating_changes) < 5:
                    continue
                
                # Calculate consistency metrics using standard deviation (lower = more consistent)
                rating_volatility = np.std(rating_changes) if rating_changes else 0      # Rating change volatility
                margin_consistency = np.std(margins) if margins else 0                   # Victory margin consistency
                win_rate = len(driver_battles[driver_battles['winnerName'] == driver]) / len(driver_battles)  # Overall win rate
                avg_rating_change = np.mean(rating_changes) if rating_changes else 0    # Average rating change per battle
                
                # Store consistency statistics for this driver
                consistency_stats[driver] = {
                    'total_battles': len(driver_battles),
                    'win_rate': win_rate,
                    'rating_volatility': rating_volatility,      # Lower = more consistent
                    'margin_consistency': margin_consistency,     # Lower = more consistent margins
                    'avg_rating_change': avg_rating_change
                }
        
        # Handle case where no drivers meet minimum battle requirement
        if not consistency_stats:
            st.info(f"📊 No drivers found with minimum {min_races} battles for consistency analysis.")
            return
        
        # Convert to DataFrame and sort by consistency (lowest volatility first)
        consistency_df = pd.DataFrame(consistency_stats).T
        consistency_df = consistency_df.sort_values('rating_volatility')  # Most consistent first
        
        # Display consistency rankings in two-column layout
        col1, col2 = st.columns(2)
        
        # Column 1: Most consistent drivers (lowest rating volatility)
        with col1:
            st.write("**Most Consistent Drivers** (Low Rating Volatility)")
            consistent_drivers = consistency_df.head(10).copy()
            consistent_drivers['Driver'] = consistent_drivers.index
            consistent_drivers['Volatility'] = consistent_drivers['rating_volatility'].round(2)
            consistent_drivers['Win Rate %'] = (consistent_drivers['win_rate'] * 100).round(1)
            consistent_drivers['Battles'] = consistent_drivers['total_battles'].astype(int)
            
            # Display consistency table
            safe_display(
                consistent_drivers[['Driver', 'Volatility', 'Win Rate %', 'Battles']],
                use_container_width=True
            )
        
        # Column 2: Most volatile drivers (highest rating volatility)
        with col2:
            st.write("**Most Volatile Drivers** (High Rating Volatility)")
            volatile_drivers = consistency_df.sort_values('rating_volatility', ascending=False).head(10).copy()
            volatile_drivers['Driver'] = volatile_drivers.index
            volatile_drivers['Volatility'] = volatile_drivers['rating_volatility'].round(2)
            volatile_drivers['Win Rate %'] = (volatile_drivers['win_rate'] * 100).round(1)
            volatile_drivers['Battles'] = volatile_drivers['total_battles'].astype(int)
            
            # Display volatility table
            safe_display(
                volatile_drivers[['Driver', 'Volatility', 'Win Rate %', 'Battles']],
                use_container_width=True
            )
        
        # Create consistency vs performance scatter plot
        if len(consistency_df) > 0:
            try:
                # Scatter plot showing relationship between consistency and performance
                fig = px.scatter(
                    consistency_df.reset_index(),
                    x='rating_volatility',     # Consistency metric (lower = more consistent)
                    y='win_rate',             # Performance metric (higher = better)
                    size='total_battles',      # Bubble size by battle experience
                    hover_name='index',        # Driver name on hover
                    title='Consistency vs Performance',
                    labels={
                        'rating_volatility': 'Rating Volatility (lower = more consistent)',
                        'win_rate': 'Win Rate'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create consistency visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in consistency analysis
        st.error(f"Error in consistency analysis: {e}")
