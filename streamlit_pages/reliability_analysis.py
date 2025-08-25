# Import required libraries for data processing, visualization, and statistical analysis
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np

# Import utility functions from local util module
from util import *


def reliability_analysis_page(history_df, rating_df):
    """
    Main function to render the F1 reliability analysis page.
    Analyzes mechanical reliability and race completion patterns through three perspectives:
    - DNF analysis (Did Not Finish rates and patterns for drivers)
    - Lap completion analysis (percentage of race distance completed)
    - Team reliability analysis (constructor-level reliability patterns and trends)
    
    Args:
        history_df: DataFrame containing historical teammate battle data with DNF information
        rating_df: DataFrame containing rating information (currently unused)
    """
    # Display page header with wrench emoji (representing mechanical aspects)
    st.header("🔧 Reliability Analysis")
    
    # **STEP 1: Add sidebar filters for safe filtering**
    # Create sidebar section for reliability-specific filters
    st.sidebar.subheader("Reliability Analysis Filters")
    
    # Year filter - multi-select with reverse chronological order (newest first)
    years = ['All'] + sorted(history_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Select Years", 
        years, 
        default=['All'],  # Default to all years
        help="Clear all = recent 10 years"  # Tooltip explaining fallback behavior
    )
    
    # Constructor/team filter - multi-select with alphabetical sorting
    constructors = ['All'] + sorted(history_df['constructor'].dropna().unique().tolist())
    selected_constructors = st.sidebar.multiselect(
        "Select Teams", 
        constructors, 
        default=['All'],  # Default to all teams
        help="Clear all = all teams"  # Tooltip explaining fallback behavior
    )
    
    # Minimum races filter - slider to ensure statistical significance for reliability metrics
    min_races = st.sidebar.slider(
        "Minimum Races per Driver/Team",
        min_value=3,   # Very low minimum for reliability analysis
        max_value=50,  # High enough for veteran drivers/teams
        value=20       # Default requires 20 races for meaningful reliability assessment
    )
    
    # **STEP 2: Apply safe filtering**
    # Apply filters using safe filtering function with reliability-specific fallback handling
    filtered_df, used_defaults = safe_filter_reliability_analysis(
        history_df, selected_years, selected_constructors
    )
    
    # Display current filtered data status in expandable section
    with st.expander("📈 Current Data Summary"):
        # Create four columns for key reliability metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total battles available for reliability analysis
        with col1:
            st.metric("Total Battles", len(filtered_df))
        
        # Column 2: Number of unique drivers in reliability analysis
        with col2:
            st.metric("Unique Drivers", pd.concat([filtered_df['winnerName'], filtered_df['loserName']]).nunique())
        
        # Column 3: Time span of reliability data
        with col3:
            st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if not filtered_df.empty else "N/A")
        
        # Column 4: Number of different teams/constructors represented
        with col4:
            st.metric("Teams", filtered_df['constructor'].nunique() if not filtered_df.empty else 0)
    
    # **STEP 3: Tabs with filtered data**
    # Create three analytical tabs, each focusing on different reliability aspects
    tab1, tab2, tab3 = st.tabs(["DNF Analysis", "Lap Completion", "Team Reliability"])
    
    # Tab 1: Analyze Did Not Finish patterns and rates for individual drivers
    with tab1:
        dnf_analysis(filtered_df, min_races)
    
    # Tab 2: Analyze lap completion rates and race distance coverage
    with tab2:
        lap_completion_analysis(filtered_df, min_races)
    
    # Tab 3: Analyze team-level reliability patterns and trends over time
    with tab3:
        team_reliability_analysis(filtered_df, min_races)


def safe_filter_reliability_analysis(history_df, selected_years, selected_constructors):
    """
    Apply safe filtering with smart defaults specifically for reliability analysis.
    Uses larger fallback datasets since reliability analysis needs extensive data for patterns.
    
    Args:
        history_df: DataFrame with historical battle data including DNF information
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
            # Default: use all available recent years for comprehensive reliability analysis
            recent_years = sorted(history_df['year'].dropna().unique(), reverse=True)
            filtered_df = filtered_df[filtered_df['year'].isin(recent_years)]
            used_defaults = True
        else:
            # Filter by user-selected years
            year_filtered = filtered_df[filtered_df['year'].isin(selected_years)]
            # Check if year filtering resulted in empty dataset
            if year_filtered.empty:
                # Fallback to recent 10 years if selection too restrictive
                recent_years = sorted(history_df['year'].dropna().unique(), reverse=True)[:10]
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
        
        # Final safety check - ensure we have adequate data for reliability analysis
        if filtered_df.empty:
            st.warning("⚠️ Filters resulted in no data. Using recent 400 battles as fallback.")
            # Use most recent 400 battles as absolute fallback (large dataset for reliability patterns)
            filtered_df = history_df.sort_values('date', ascending=False).head(400)
            used_defaults = True
        
        # Ensure minimum data for meaningful reliability analysis
        # Reliability analysis requires substantial data for meaningful failure patterns
        if len(filtered_df) < 50:
            st.info("📊 Limited data with current filters. Expanding to show more results...")
            # Get more historical data for better reliability pattern analysis
            recent_data = history_df.sort_values('date', ascending=False).head(800)
            if len(recent_data) > len(filtered_df):
                filtered_df = recent_data
                used_defaults = True
    
    except Exception as e:
        # Handle any unexpected errors in filtering process
        st.error(f"Error in filtering: {e}")
        # Use last 400 battles as emergency fallback (large dataset for reliability analysis)
        filtered_df = history_df.tail(400)
        used_defaults = True
    
    return filtered_df, used_defaults


# **UPDATED ANALYSIS FUNCTIONS WITH SAFE HANDLING**


def dnf_analysis(history_df, min_races=5):
    """
    Analyze Did Not Finish (DNF) patterns for individual drivers.
    DNFs indicate mechanical failures, crashes, or other race-ending incidents.
    Categorizes DNFs by timing (early vs late) to identify different failure patterns.
    
    Args:
        history_df: DataFrame containing battle history with DNF status information
        min_races: Minimum number of races required for inclusion in analysis (default: 5)
    """
    # Display section header with X emoji (representing DNF/failure)
    st.subheader("❌ DNF Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for DNF analysis")
        return
    
    try:
        # Calculate DNF statistics for each driver with comprehensive safety handling
        dnf_stats = {}
        # Get all unique drivers from both winner and loser columns
        all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
        
        # Process each driver's DNF patterns
        for driver in all_drivers:
            if pd.isna(driver):
                continue
                
            # Get all races where this driver participated
            driver_races = history_df[
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            ]
            
            total_races = len(driver_races)
            # Only analyze drivers with sufficient race data for statistical significance
            if total_races >= min_races:
                # Initialize DNF counters for different failure timing categories
                dnf_count = 0          # Total DNFs
                early_dnf_count = 0    # DNFs in first 25% of race (likely mechanical/setup issues)
                late_dnf_count = 0     # DNFs in last 25% of race (likely wear-related failures)
                
                # Process each race to categorize DNF types
                for _, race in driver_races.iterrows():
                    try:
                        # Check if this driver won or lost the battle to get correct DNF status
                        if race['winnerName'] == driver:
                            dnf_status = pd.to_numeric(race['dnf_W'], errors='coerce')
                            if dnf_status == 1:  # DNF occurred
                                dnf_count += 1
                                # Safe completion rate calculation for DNF timing classification
                                laps_completed = pd.to_numeric(race['lapsCompleted_W'], errors='coerce')
                                race_distance = pd.to_numeric(race['race_distance'], errors='coerce')
                                
                                # Categorize DNF by race completion percentage
                                if pd.notna(laps_completed) and pd.notna(race_distance) and race_distance > 0:
                                    completion_rate = laps_completed / race_distance
                                    if completion_rate < 0.25:      # Failed in first quarter
                                        early_dnf_count += 1
                                    elif completion_rate > 0.75:    # Failed in last quarter
                                        late_dnf_count += 1
                        else:
                            # Driver was the loser in this battle
                            dnf_status = pd.to_numeric(race['dnf_L'], errors='coerce')
                            if dnf_status == 1:  # DNF occurred
                                dnf_count += 1
                                # Safe completion rate calculation for DNF timing classification
                                laps_completed = pd.to_numeric(race['lapsCompleted_L'], errors='coerce')
                                race_distance = pd.to_numeric(race['race_distance'], errors='coerce')
                                
                                # Categorize DNF by race completion percentage
                                if pd.notna(laps_completed) and pd.notna(race_distance) and race_distance > 0:
                                    completion_rate = laps_completed / race_distance
                                    if completion_rate < 0.25:      # Failed in first quarter
                                        early_dnf_count += 1
                                    elif completion_rate > 0.75:    # Failed in last quarter
                                        late_dnf_count += 1
                    except Exception:
                        continue  # Skip problematic race data
                
                # Calculate comprehensive DNF statistics for this driver
                dnf_stats[driver] = {
                    'total_races': total_races,
                    'dnf_count': dnf_count,
                    'dnf_rate': dnf_count / total_races,                    # Overall DNF percentage
                    'early_dnf_rate': early_dnf_count / total_races,       # Early failure rate
                    'late_dnf_rate': late_dnf_count / total_races,         # Late failure rate
                    'reliability_score': 1 - (dnf_count / total_races)     # Reliability score (higher = more reliable)
                }
        
        # Handle case where no drivers meet minimum race requirement
        if not dnf_stats:
            st.info(f"📊 No drivers found with minimum {min_races} races for DNF analysis.")
            return
        
        # Convert statistics to DataFrame for analysis and display
        dnf_df = pd.DataFrame(dnf_stats).T
        # Sort by reliability score (most reliable drivers first)
        dnf_df = dnf_df.sort_values('reliability_score', ascending=False)
        
        # Display reliability rankings in two-column layout
        col1, col2 = st.columns(2)
        
        # Column 1: Most reliable drivers (lowest DNF rates)
        with col1:
            st.write("**Most Reliable Drivers** (Lowest DNF Rate)")
            reliable_drivers = dnf_df.head(10).copy()
            reliable_drivers['Driver'] = reliable_drivers.index
            reliable_drivers['DNF Rate %'] = (reliable_drivers['dnf_rate'] * 100).round(1)
            reliable_drivers['Reliability Score'] = (reliable_drivers['reliability_score'] * 100).round(1)
            
            # Display most reliable drivers table
            safe_display(
                reliable_drivers[['Driver', 'total_races', 'dnf_count', 'DNF Rate %']].rename(columns={
                    'total_races': 'Races',
                    'dnf_count': 'DNFs'
                }),
                use_container_width=True
            )
        
        # Column 2: Drivers with highest DNF rates (least reliable)
        with col2:
            st.write("**Highest DNF Rates**")
            unreliable_drivers = dnf_df.sort_values('dnf_rate', ascending=False).head(10).copy()
            unreliable_drivers['Driver'] = unreliable_drivers.index
            unreliable_drivers['DNF Rate %'] = (unreliable_drivers['dnf_rate'] * 100).round(1)
            
            # Display highest DNF rate drivers table
            safe_display(
                unreliable_drivers[['Driver', 'total_races', 'dnf_count', 'DNF Rate %']].rename(columns={
                    'total_races': 'Races',
                    'dnf_count': 'DNFs'
                }),
                use_container_width=True
            )
        
        # Analyze DNF timing patterns across all drivers
        st.subheader("DNF Timing Analysis")
        
        # Calculate average DNF rates by timing category across all drivers
        avg_early_dnf = dnf_df['early_dnf_rate'].mean() * 100      # Early DNF rate (0-25% race completion)
        avg_late_dnf = dnf_df['late_dnf_rate'].mean() * 100        # Late DNF rate (75-100% race completion)
        avg_total_dnf = dnf_df['dnf_rate'].mean() * 100            # Overall DNF rate
        
        # Display DNF timing statistics in three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Early DNF Rate", f"{avg_early_dnf:.1f}%", "0-25% race distance")
        with col2:
            st.metric("Avg Late DNF Rate", f"{avg_late_dnf:.1f}%", "75-100% race distance")
        with col3:
            st.metric("Avg Total DNF Rate", f"{avg_total_dnf:.1f}%")
        
        # Create DNF rate visualization with error handling
        if len(dnf_df) > 0:
            try:
                # Scatter plot showing relationship between race experience and DNF rate
                fig = px.scatter(
                    dnf_df.reset_index(),
                    x='total_races',      # Experience level (number of races)
                    y='dnf_rate',         # DNF rate (reliability measure)
                    size='dnf_count',     # Bubble size by total DNF count
                    hover_name='index',   # Driver name on hover
                    title='DNF Rate vs Race Experience',
                    labels={
                        'total_races': 'Total Races',
                        'dnf_rate': 'DNF Rate'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create DNF visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in DNF analysis
        st.error(f"Error in DNF analysis: {e}")


def lap_completion_analysis(history_df, min_races=5):
    """
    Analyze lap completion patterns and race distance coverage for drivers.
    Measures what percentage of each race distance drivers typically complete.
    Higher completion rates indicate better mechanical reliability and fewer failures.
    
    Args:
        history_df: DataFrame containing battle history with lap completion data
        min_races: Minimum number of races required for inclusion in analysis (default: 5)
    """
    # Display section header with checkered flag emoji
    st.subheader("🏁 Lap Completion Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for lap completion analysis")
        return
    
    try:
        # Calculate lap completion statistics for each driver with comprehensive safety handling
        lap_stats = {}
        # Get all unique drivers from both winner and loser columns
        all_drivers = pd.concat([history_df['winnerName'], history_df['loserName']]).dropna().unique()
        
        # Process each driver's lap completion patterns
        for driver in all_drivers:
            if pd.isna(driver):
                continue
                
            # Get all races where this driver participated
            driver_races = history_df[
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            ]
            
            # Only analyze drivers with sufficient race data for statistical significance
            if len(driver_races) >= min_races:
                completion_rates = []    # List to store completion rate for each race
                total_laps = 0          # Running total of laps completed
                total_possible_laps = 0 # Running total of possible laps
                
                # Process each race to calculate completion statistics
                for _, race in driver_races.iterrows():
                    try:
                        # Get race distance (total laps in the race)
                        race_distance = pd.to_numeric(race['race_distance'], errors='coerce')
                        
                        # Determine laps completed based on whether driver won or lost battle
                        if race['winnerName'] == driver:
                            laps_completed = pd.to_numeric(race['lapsCompleted_W'], errors='coerce')
                        else:
                            laps_completed = pd.to_numeric(race['lapsCompleted_L'], errors='coerce')
                        
                        # Only process races with valid lap data
                        if pd.notna(race_distance) and pd.notna(laps_completed) and race_distance > 0:
                            # Calculate completion rate for this race (0.0 to 1.0)
                            completion_rate = laps_completed / race_distance
                            completion_rates.append(completion_rate)
                            total_laps += laps_completed
                            total_possible_laps += race_distance
                    except Exception:
                        continue  # Skip problematic race data
                
                # Calculate comprehensive lap completion statistics if we have valid data
                if len(completion_rates) > 0:
                    lap_stats[driver] = {
                        'total_races': len(driver_races),
                        'avg_completion_rate': np.mean(completion_rates),              # Average completion per race
                        'completion_consistency': 1 - np.std(completion_rates) if len(completion_rates) > 1 else 1,  # Consistency score
                        'total_laps': total_laps,                                      # Total laps completed across all races
                        'total_possible_laps': total_possible_laps,                    # Total possible laps across all races
                        'overall_completion': total_laps / total_possible_laps if total_possible_laps > 0 else 0  # Overall completion rate
                    }
        
        # Handle case where no drivers have sufficient lap completion data
        if not lap_stats:
            st.info(f"📊 No drivers found with sufficient lap completion data (minimum {min_races} races).")
            return
        
        # Convert statistics to DataFrame for analysis and display
        lap_df = pd.DataFrame(lap_stats).T
        # Sort by average completion rate (best completion rates first)
        lap_df = lap_df.sort_values('avg_completion_rate', ascending=False)
        
        # Display lap completion leaders
        st.write("**Best Lap Completion Rates**")
        
        # Prepare display-friendly format with proper rounding and formatting
        completion_display = lap_df.head(15).copy()  # Top 15 for comprehensive view
        completion_display['Driver'] = completion_display.index
        completion_display['Avg Completion %'] = (completion_display['avg_completion_rate'] * 100).round(1)
        completion_display['Overall Completion %'] = (completion_display['overall_completion'] * 100).round(1)
        completion_display['Total Laps'] = completion_display['total_laps'].astype(int)
        completion_display['Total Possible Laps'] = completion_display['total_possible_laps'].astype(int)
        completion_display['Consistency Score'] = (completion_display['completion_consistency'] * 100).round(1)
        
        # Display comprehensive lap completion table
        safe_display(
            completion_display[['Driver', 'total_races', 'Avg Completion %', 'Overall Completion %', 'Total Laps', 'Total Possible Laps', 'Consistency Score']].rename(columns={
                'total_races': 'Races'
            }),
            use_container_width=True
        )
        
        # Create lap completion visualization with error handling
        if len(lap_df) > 0:
            try:
                # Scatter plot showing relationship between completion rate and consistency
                fig = px.scatter(
                    lap_df.reset_index(),
                    x='avg_completion_rate',      # Average completion rate per race
                    y='completion_consistency',   # Consistency of completion (lower variance = higher consistency)
                    size='total_races',           # Bubble size by race experience
                    hover_name='index',           # Driver name on hover
                    title='Lap Completion: Rate vs Consistency',
                    labels={
                        'avg_completion_rate': 'Average Completion Rate',
                        'completion_consistency': 'Completion Consistency'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                st.warning(f"Could not create lap completion visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in lap completion analysis
        st.error(f"Error in lap completion analysis: {e}")


def team_reliability_analysis(history_df, min_races=10):
    """
    Analyze team/constructor reliability patterns and trends over time.
    Shows which teams have the most reliable cars and how reliability has changed.
    Tracks DNF rates at the constructor level and identifies reliability improvements/declines.
    
    Args:
        history_df: DataFrame containing battle history with constructor and DNF information
        min_races: Minimum number of races required for team inclusion (default: 10)
    """
    # Display section header with race car emoji
    st.subheader("🏎️ Team Reliability Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for team reliability analysis")
        return
    
    try:
        # Calculate team reliability statistics with comprehensive safety handling
        team_stats = {}
        # Get all unique constructors/teams from the dataset
        available_constructors = history_df['constructor'].dropna().unique()
        
        # Process each team's reliability patterns
        for constructor in available_constructors:
            if pd.isna(constructor):
                continue
                
            # Get all races for this constructor/team
            team_races = history_df[history_df['constructor'] == constructor]
            
            # Only analyze teams with sufficient race data for statistical significance
            if len(team_races) >= min_races:
                # Each race has 2 drivers, so total driver-races = races * 2
                total_driver_races = len(team_races) * 2
                
                # Safe DNF calculation for both drivers in each race
                dnf_w = pd.to_numeric(team_races['dnf_W'], errors='coerce').sum()  # Winner DNFs
                dnf_l = pd.to_numeric(team_races['dnf_L'], errors='coerce').sum()  # Loser DNFs
                total_dnfs = dnf_w + dnf_l
                
                # Calculate yearly DNF rates for trend analysis
                yearly_dnf_rates = {}
                years = pd.to_numeric(team_races['year'], errors='coerce').dropna().unique()
                
                # Process each year to track reliability trends over time
                for year in sorted(years):
                    year_races = team_races[pd.to_numeric(team_races['year'], errors='coerce') == year]
                    if len(year_races) > 0:
                        year_driver_races = len(year_races) * 2  # 2 drivers per race
                        year_dnf_w = pd.to_numeric(year_races['dnf_W'], errors='coerce').sum()
                        year_dnf_l = pd.to_numeric(year_races['dnf_L'], errors='coerce').sum()
                        year_dnfs = year_dnf_w + year_dnf_l
                        # Calculate DNF rate for this year
                        yearly_dnf_rates[int(year)] = year_dnfs / year_driver_races if year_driver_races > 0 else 0
                
                # Store comprehensive team reliability statistics
                team_stats[constructor] = {
                    'total_races': len(team_races),
                    'total_dnfs': int(total_dnfs),
                    'dnf_rate': total_dnfs / total_driver_races if total_driver_races > 0 else 0,  # Overall DNF rate
                    'yearly_rates': yearly_dnf_rates,                                               # Year-by-year trends
                    'recent_dnf_rate': list(yearly_dnf_rates.values())[-3:] if len(yearly_dnf_rates) >= 3 else []  # Recent 3 years
                }
        
        # Handle case where no teams meet minimum race requirement
        if not team_stats:
            st.info(f"📊 No teams found with minimum {min_races} races for reliability analysis.")
            return
        
        # Convert team statistics to DataFrame for analysis and display
        team_df = pd.DataFrame({
            constructor: {
                'total_races': stats['total_races'],
                'dnf_rate': stats['dnf_rate'],
                'total_dnfs': stats['total_dnfs']
            }
            for constructor, stats in team_stats.items()
        }).T
        
        # Sort by DNF rate (most reliable teams first)
        team_df = team_df.sort_values('dnf_rate')
        
        # Display team reliability rankings in two-column layout
        col1, col2 = st.columns(2)
        
        # Column 1: Most reliable teams (lowest DNF rates)
        with col1:
            st.write("**Most Reliable Teams** (Lowest DNF Rate)")
            reliable_teams = team_df.head(8).copy()  # Top 8 most reliable
            reliable_teams['Team'] = reliable_teams.index
            reliable_teams['DNF Rate %'] = (reliable_teams['dnf_rate'] * 100).round(1)
            reliable_teams['Reliability Score'] = ((1 - reliable_teams['dnf_rate']) * 100).round(1)
            
            # Display most reliable teams table
            safe_display(
                reliable_teams[['Team', 'total_races', 'total_dnfs', 'DNF Rate %', 'Reliability Score']].rename(columns={
                    'total_races': 'Races',
                    'total_dnfs': 'DNFs'
                }),
                use_container_width=True
            )
        
        # Column 2: Least reliable teams (highest DNF rates)
        with col2:
            st.write("**Least Reliable Teams**")
            unreliable_teams = team_df.sort_values('dnf_rate', ascending=False).head(8).copy()  # Top 8 least reliable
            unreliable_teams['Team'] = unreliable_teams.index
            unreliable_teams['DNF Rate %'] = (unreliable_teams['dnf_rate'] * 100).round(1)
            
            # Display least reliable teams table
            safe_display(
                unreliable_teams[['Team', 'total_races', 'total_dnfs', 'DNF Rate %']].rename(columns={
                    'total_races': 'Races', 
                    'total_dnfs': 'DNFs'
                }),
                use_container_width=True
            )
        
        # Analyze team reliability trends over time
        st.subheader("Team Reliability Trends")
        
        # Get teams with sufficient historical data for trend analysis
        available_teams = [team for team in team_stats.keys()]# if len(team_stats[team]['yearly_rates']) > 1]

        if available_teams:
            # Pre-select major F1 teams for default trend display
            # default_teams = list(team_df.head(5).index) if len(team_df) >= 5 else available_teams[:5]
            default_teams = ["Ferrari", "Red Bull", "Mercedes", "McLaren", "Williams"]
            
            # Multi-select widget for choosing teams to display in trend analysis
            selected_teams = st.multiselect(
                "Select Teams for Trend Analysis",
                options=available_teams,
                default=default_teams
            )
            
            # Create and display reliability trend chart if teams are selected
            if selected_teams:
                trend_data = []
                # Build trend data for selected teams
                for team in selected_teams:
                    if team in team_stats:
                        # Add each year's DNF rate for this team
                        for year, dnf_rate in team_stats[team]['yearly_rates'].items():
                            trend_data.append({
                                'Team': team,
                                'Year': int(year),
                                'DNF Rate %': dnf_rate * 100  # Convert to percentage
                            })
                
                # Create trend visualization if we have data
                if trend_data:
                    try:
                        trend_df = pd.DataFrame(trend_data)
                        # Line chart showing how each team's reliability has changed over time
                        fig = px.line(
                            trend_df,
                            x='Year',
                            y='DNF Rate %',
                            color='Team',          # Different color for each team
                            title='Team Reliability Trends Over Time',
                            markers=True           # Show data points
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as viz_error:
                        # Handle visualization errors gracefully
                        st.warning(f"Could not create team trends visualization: {viz_error}")
        else:
            st.info("Not enough historical data for team reliability trends")
    
    except Exception as e:
        # Handle any errors in team reliability analysis
        st.error(f"Error in team reliability analysis: {e}")
