# Import required libraries for data processing, visualization, and statistical analysis
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go

# Import utility functions from local util module
from util import *


def race_statistics_page(history_df, rating_df, result_df):
    """
    Main function to render the F1 race statistics page.
    Provides comprehensive race-focused analysis through three analytical perspectives:
    - Grand Prix records (track-specific performance and historical records)
    - Season analysis (championship progression and yearly statistics)
    - Race characteristics (general racing patterns and trends over time)
    
    Args:
        history_df: DataFrame containing historical teammate battle data
        rating_df: DataFrame containing rating information (currently unused)
        result_df: DataFrame containing race results and final positions
    """
    # Display page header with checkered flag emoji
    st.header("🏁 Race Statistics")
    
    # **STEP 1: Add sidebar filters for safe filtering**
    # Create sidebar section for race-specific filters
    st.sidebar.subheader("Race Statistics Filters")
    
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
    
    # **STEP 2: Apply safe filtering**
    # Apply filters using safe filtering function with race-specific fallback handling
    filtered_df, used_defaults = safe_filter_race_statistics(
        history_df, selected_years, selected_constructors
    )
    
    # Display current filtered data status in expandable section
    with st.expander("📈 Current Data Summary"):
        # Create four columns for key race statistics metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total teammate battles available for analysis
        with col1:
            st.metric("Total Battles", len(filtered_df))
        
        # Column 2: Number of unique Grand Prix races represented
        with col2:
            st.metric("Unique Races", filtered_df['raceName'].nunique() if not filtered_df.empty else 0)
        
        # Column 3: Time span of race data available
        with col3:
            st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if not filtered_df.empty else "N/A")
        
        # Column 4: Number of different teams/constructors represented
        with col4:
            st.metric("Teams", filtered_df['constructor'].nunique() if not filtered_df.empty else 0)
    
    # **STEP 3: Tabs with filtered data**
    # Create three analytical tabs, each focusing on different race-related aspects
    tab1, tab2, tab3 = st.tabs(["Grand Prix Records", "Season Analysis", "Race Characteristics"])
    
    # Tab 1: Analyze track-specific records and Grand Prix history
    with tab1:
        grand_prix_records(filtered_df, result_df)
    
    # Tab 2: Analyze season progression and championship battles
    with tab2:
        season_analysis(filtered_df, result_df)
    
    # Tab 3: Analyze general racing patterns and characteristics
    with tab3:
        race_characteristics(filtered_df)


def safe_filter_race_statistics(history_df, selected_years, selected_constructors):
    """
    Apply safe filtering with smart defaults specifically for race statistics analysis.
    Uses larger fallback datasets since race analysis needs comprehensive historical data.
    
    Args:
        history_df: DataFrame with historical race battle data
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
            # Default: use all available years for comprehensive race analysis
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
        
        # Final safety check - ensure we have adequate data for race analysis
        if filtered_df.empty:
            st.warning("⚠️ Filters resulted in no data. Using recent 300 battles as fallback.")
            # Use most recent 300 battles as absolute fallback (larger dataset for race analysis)
            filtered_df = history_df.sort_values('date', ascending=False).head(300)
            used_defaults = True
        
        # Ensure minimum data for meaningful race statistics analysis
        # Race statistics requires substantial data for meaningful track and season patterns
        if len(filtered_df) < 50:
            st.info("📊 Limited data with current filters. Expanding to show more results...")
            # Get more historical data for better race pattern analysis
            recent_data = history_df.sort_values('date', ascending=False).head(1000)
            if len(recent_data) > len(filtered_df):
                filtered_df = recent_data
                used_defaults = True
    
    except Exception as e:
        # Handle any unexpected errors in filtering process
        st.error(f"Error in filtering: {e}")
        # Use last 500 battles as emergency fallback (large dataset for race analysis)
        filtered_df = history_df.tail(500)
        used_defaults = True
    
    return filtered_df, used_defaults


# **UPDATED ANALYSIS FUNCTIONS WITH SAFE HANDLING**


def grand_prix_records(history_df, result_df):
    """
    Analyze Grand Prix specific records and track-based performance patterns.
    Shows most successful drivers at each circuit, drama factors, and track characteristics.
    Provides both overview of all tracks and detailed analysis of selected circuits.
    
    Args:
        history_df: DataFrame containing battle history with race names and track data
        result_df: DataFrame containing race results with final positions
    """
    # Display section header with trophy emoji
    st.subheader("🏆 Grand Prix Records")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No race data available for Grand Prix analysis")
        return
    
    # Race selection interface with comprehensive validation
    try:
        # Get all unique race names from the dataset
        available_races = history_df['raceName'].dropna().unique()
        if len(available_races) == 0:
            st.warning("No race names found in data")
            return
        
        # Create dropdown with 'All' option plus sorted race names
        races = ['All'] + sorted(available_races.tolist())
        selected_race = st.selectbox("Select Grand Prix", races)
        
        # Handle 'All' selection - show summary for all Grand Prix
        if selected_race == 'All':
            # Show comprehensive summary for all races with safe handling
            gp_summary = []

            # Merge with result data to get actual race winners (finalPosition == 1)
            result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
            result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

            # Join battle history with race results
            history_added_df = pd.merge(history_df, result_filtered_df, on=["year", "round", "winnerId"])

            # Filter for actual race winners only
            history_added_df = history_added_df[history_added_df["finalPosition"]==1]
            
            # Process each Grand Prix to build summary statistics
            for race_name in available_races:
                try:
                    # Get all data for this specific Grand Prix
                    race_data = history_added_df[history_added_df['raceName'] == race_name]
                    
                    # Skip races with no data
                    if race_data.empty:
                        continue
                    
                    # Find most successful driver with safe handling
                    winner_counts = race_data['winnerName'].value_counts()
                    most_successful = winner_counts.index[0] if len(winner_counts) > 0 else 'N/A'
                    most_wins = int(winner_counts.iloc[0]) if len(winner_counts) > 0 else 0  # **FIXED BUG** - Convert to int
                    
                    # Calculate drama factor (average Elo rating change) with safe calculation
                    drama_factor = pd.to_numeric(race_data['delta'], errors='coerce').mean()
                    drama_factor = drama_factor if pd.notna(drama_factor) else 0
                    
                    # Determine years this Grand Prix was active with safe handling
                    years = pd.to_numeric(race_data['year'], errors='coerce').dropna()
                    years_active = f"{int(years.min())}-{int(years.max())}" if len(years) > 0 else 'Unknown'
                    
                    # Add Grand Prix summary to results
                    gp_summary.append({
                        'Grand Prix': race_name,
                        'Most Successful': most_successful,
                        'Wins': int(most_wins),
                        'Drama Factor': f"{drama_factor:.1f}",  # Format to 1 decimal place
                        'Total Battles': len(race_data),
                        'Years': years_active
                    })
                except Exception as e:
                    # Log processing errors but continue with other races
                    st.warning(f"Error processing {race_name}: {e}")
                    continue
            
            # Display Grand Prix summary table if data available
            if gp_summary:
                summary_df = pd.DataFrame(gp_summary)
                # Safe sorting by Drama Factor (most dramatic races first)
                try:
                    summary_df['Drama Factor Numeric'] = pd.to_numeric(summary_df['Drama Factor'], errors='coerce')
                    summary_df = summary_df.sort_values('Drama Factor Numeric', ascending=False).drop('Drama Factor Numeric', axis=1)
                except Exception:
                    pass  # Keep original order if sorting fails
                
                safe_display(summary_df, use_container_width=True)
            else:
                st.info("No Grand Prix data available for summary")
        
        else:
            # Detailed analysis for a specific Grand Prix with safe handling
            race_data = history_df[history_df['raceName'] == selected_race]
            
            # Check if selected race has data
            if race_data.empty:
                st.warning(f"No data available for {selected_race}")
                return
            
            # Display Grand Prix statistics in four-column layout with safe calculations
            col1, col2, col3, col4 = st.columns(4)
            
            # Column 1: Total battles at this Grand Prix
            with col1:
                total_battles = len(race_data)
                st.metric("Total Battles", total_battles)
            
            # Column 2: Years this Grand Prix has been active
            with col2:
                try:
                    # Safe year range calculation
                    years = pd.to_numeric(race_data['year'], errors='coerce').dropna()
                    years_span = f"{int(years.min())}-{int(years.max())}" if len(years) > 0 else "Unknown"
                    st.metric("Years Active", years_span)
                except Exception:
                    st.metric("Years Active", "Unknown")
            
            # Column 3: Drama factor (average rating change magnitude)
            with col3:
                try:
                    # Calculate average Elo rating change as measure of race excitement
                    avg_drama = pd.to_numeric(race_data['delta'], errors='coerce').mean()
                    st.metric("Drama Factor", f"{avg_drama:.1f}" if pd.notna(avg_drama) else "N/A")
                except Exception:
                    st.metric("Drama Factor", "N/A")
            
            # Column 4: Comeback rate (percentage of wins from grid deficit)
            with col4:
                try:
                    # Calculate percentage of wins where driver overcame grid position deficit
                    startDiff_data = pd.to_numeric(race_data['startDiff'], errors='coerce')
                    comeback_battles = (startDiff_data > 2).sum()  # Grid deficit > 2 positions
                    comeback_rate = comeback_battles / len(race_data) * 100
                    st.metric("Comeback Rate", f"{comeback_rate:.1f}%")
                except Exception:
                    st.metric("Comeback Rate", "N/A")
            
            # Analyze most successful drivers at this specific Grand Prix
            st.subheader(f"Most Successful Drivers at {selected_race}")
            
            try:
                # Merge race data with results to get actual race winners
                result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
                result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

                # Join battle data with race results
                race_data = pd.merge(race_data, result_filtered_df, on=["year", "round", "winnerId"])

                # Analyze actual race winners (finalPosition == 1)
                winner_analysis = race_data[race_data["finalPosition"]==1]['winnerName'].value_counts().head(10)
                
                if not winner_analysis.empty:
                    # Create winners summary table
                    winner_df = pd.DataFrame({
                        'Driver': winner_analysis.index,
                        'Wins': winner_analysis.values,
                    })
                    
                    # Calculate win rate at this track safely
                    try:
                        # Get total races per driver at this track
                        driver_total_races = race_data.groupby('winnerName').size().reindex(winner_analysis.index)
                        winner_df['Win Rate %'] = (winner_analysis.values / driver_total_races.values * 100).round(1)
                    except Exception:
                        winner_df['Win Rate %'] = 'N/A'  # Fallback if calculation fails
                    
                    # Display winners table
                    safe_display(winner_df, use_container_width=True)
                    
                    # Create wins visualization with error handling
                    try:
                        # Bar chart showing most successful drivers at this track
                        fig = px.bar(
                            winner_df.head(8),  # Top 8 for readability
                            x='Driver',
                            y='Wins',
                            title=f'Most Wins at {selected_race}'
                        )
                        fig.update_xaxes(tickangle=45)  # Rotate driver names for readability
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as viz_error:
                        st.warning(f"Could not create wins visualization: {viz_error}")
                else:
                    st.info("No winner data available")
            except Exception as e:
                st.error(f"Error analyzing winners: {e}")
            
            # Show recent winners at this Grand Prix with safe handling
            st.subheader("Recent Winners")
            try:
                # Get recent race winners sorted by date (most recent first)
                recent_winners = race_data[race_data["finalPosition"]==1].sort_values('date', ascending=False)
                
                if not recent_winners.empty:
                    recent_display = recent_winners.copy()
                    
                    # Safe numeric formatting for display metrics
                    if 'margin' in recent_display.columns:
                        recent_display['Victory Margin'] = pd.to_numeric(recent_display['margin'], errors='coerce').round(2)
                    if 'startDiff' in recent_display.columns:
                        recent_display['Grid Deficit'] = pd.to_numeric(recent_display['startDiff'], errors='coerce').astype('Int64')
                    if 'delta' in recent_display.columns:
                        recent_display['Rating Gain'] = pd.to_numeric(recent_display['delta'], errors='coerce').round(1)
                    
                    # Select available columns for display (only show what exists)
                    base_columns = ['year', 'winnerName']
                    optional_columns = ['Victory Margin', 'Grid Deficit', 'Rating Gain']
                    display_columns = base_columns + [col for col in optional_columns if col in recent_display.columns]
                    
                    # Display recent winners table with renamed columns
                    safe_display(
                        recent_display[display_columns].rename(columns={
                            'year': 'Year',
                            'winnerName': 'Winner'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No recent winner data available")
            except Exception as e:
                st.error(f"Error displaying recent winners: {e}")
    
    except Exception as e:
        # Handle any errors in Grand Prix records analysis
        st.error(f"Error in Grand Prix records analysis: {e}")


def season_analysis(history_df, result_df):
    """
    Analyze season progression and championship battles year by year.
    Shows championship progression, season statistics, and race-by-race winner breakdown.
    Focuses on how championships develop over the course of a season.
    
    Args:
        history_df: DataFrame containing battle history with season data
        result_df: DataFrame containing race results with final positions
    """
    # Display section header with trending chart emoji
    st.subheader("📈 Season Analysis")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for season analysis")
        return
    
    try:
        # Merge battle history with race results to focus on actual race winners
        result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
        result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

        # Join battle data with race results
        history_df = pd.merge(history_df, result_filtered_df, on=["year", "round", "winnerId"])
        # Filter for actual race winners only
        history_df = history_df[history_df["finalPosition"]==1]

        # Year selection interface with validation
        available_years = pd.to_numeric(history_df['year'], errors='coerce').dropna().unique()

        # Check if we have valid year data
        if len(available_years) == 0:
            st.warning("No valid year data found")
            return
        
        # Create year selection dropdown (newest years first)
        years = sorted([int(year) for year in available_years], reverse=True)
        selected_year = st.selectbox("Select Season", years)
        
        # Filter data for selected season
        season_data = history_df[pd.to_numeric(history_df['year'], errors='coerce') == selected_year]
        
        # Check if selected season has data
        if season_data.empty:
            st.warning(f"No data for {selected_year} season")
            return
        
        # Display season overview metrics in four-column layout with safe calculations
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total races in the season
        with col1:
            try:
                # Get highest round number as total races
                total_races = int(pd.to_numeric(season_data['round'], errors='coerce').max())
                st.metric("Total Races", total_races)
            except Exception:
                # Fallback to unique battle count
                st.metric("Total Races", len(season_data))
        
        # Column 2: Number of different race winners
        with col2:
            unique_winners = season_data['winnerName'].nunique()
            st.metric("Different Winners", unique_winners)
        
        # Column 3: Season drama factor (average rating change)
        with col3:
            try:
                # Calculate average Elo rating change as measure of season excitement
                avg_drama = pd.to_numeric(season_data['delta'], errors='coerce').mean()
                st.metric("Season Drama", f"{avg_drama:.1f}" if pd.notna(avg_drama) else "N/A")
            except Exception:
                st.metric("Season Drama", "N/A")
        
        # Column 4: Most successful driver of the season
        with col4:
            try:
                # Find driver with most wins in the season
                winner_counts = season_data['winnerName'].value_counts()
                dominant_driver = winner_counts.index[0] if len(winner_counts) > 0 else "N/A"
                dominant_wins = winner_counts.iloc[0] if len(winner_counts) > 0 else 0  # **FIXED BUG** - Convert to int
                st.metric("Most Wins", f"{dominant_driver[:15]}", f"{dominant_wins} wins")  # Truncate long names
            except Exception:
                st.metric("Most Wins", "N/A", "No data")
        
        # Create championship progression visualization with safe handling
        st.subheader(f"{selected_year} Championship Progression")
        
        try:
            # Build championship progression data (cumulative wins by round)
            championship_data = []
            drivers = season_data['winnerName'].dropna().unique()
            rounds = pd.to_numeric(season_data['round'], errors='coerce').dropna().unique()
            rounds = sorted([int(r) for r in rounds])
            
            # Calculate cumulative wins for each driver after each round
            for round_num in rounds:
                # Get all races up to this round
                round_data = season_data[pd.to_numeric(season_data['round'], errors='coerce') <= round_num]
                wins_so_far = round_data['winnerName'].value_counts()
                
                # Record wins for each driver at this point in season
                for driver in drivers:
                    wins = wins_so_far.get(driver, 0)
                    championship_data.append({
                        'Round': round_num,
                        'Driver': driver,
                        'Wins': wins
                    })
            
            if championship_data:
                championship_df = pd.DataFrame(championship_data)
                
                # Create championship battle line chart for top contenders
                top_drivers = season_data['winnerName'].value_counts().head(6).index.tolist()  # Top 6 for readability
                championship_top = championship_df[championship_df['Driver'].isin(top_drivers)]
                
                if not championship_top.empty:
                    # Line chart showing cumulative wins progression
                    fig = px.line(
                        championship_top,
                        x='Round',
                        y='Wins',
                        color='Driver',
                        title=f'{selected_year} Championship Battle - Cumulative Wins',
                        markers=True  # Show data points
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as viz_error:
            st.warning(f"Could not create championship progression chart: {viz_error}")
        
        # Display season winners table with safe handling
        st.subheader(f"{selected_year} Race Winners")
        try:
            # Get season winners sorted by round number
            season_winners = season_data.sort_values('round').copy()
            
            # Safe column selection and formatting
            base_columns = ['round', 'raceName', 'winnerName']
            available_base = [col for col in base_columns if col in season_winners.columns]
            
            if available_base:
                display_df = season_winners[available_base].copy()
                
                # Add optional performance metrics if available
                if 'margin' in season_winners.columns:
                    display_df['Victory Margin'] = pd.to_numeric(season_winners['margin'], errors='coerce').round(2)
                if 'delta' in season_winners.columns:
                    display_df['Rating Gain'] = pd.to_numeric(season_winners['delta'], errors='coerce').round(1)
                
                # Rename columns for better presentation
                column_mapping = {'round': 'Round', 'raceName': 'Race', 'winnerName': 'Winner'}
                display_df = display_df.rename(columns=column_mapping)
                
                # Display season winners table
                safe_display(display_df, use_container_width=True)
            else:
                st.info("Race winner data not available")
        except Exception as e:
            st.error(f"Error displaying season winners: {e}")
    
    except Exception as e:
        # Handle any errors in season analysis
        st.error(f"Error in season analysis: {e}")


def race_characteristics(history_df):
    """
    Analyze general racing characteristics and patterns across all data.
    Shows overall statistics, rating distributions, correlations, and era trends.
    Provides insights into how racing has evolved over time.
    
    Args:
        history_df: DataFrame containing battle history with performance metrics
    """
    # Display section header with checkered flag emoji
    st.subheader("🏁 Race Characteristics")
    
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for race characteristics analysis")
        return
    
    try:
        # Display overall racing statistics in four-column layout with safe calculations
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total battles in dataset
        with col1:
            total_battles = len(history_df)
            st.metric("Total Battles", f"{total_battles:,}")  # Format with thousands separator
        
        # Column 2: Average victory margin
        with col2:
            try:
                # Calculate average margin of victory across all battles
                avg_margin = pd.to_numeric(history_df['margin'], errors='coerce').mean()
                st.metric("Avg Victory Margin", f"{avg_margin:.2f}" if pd.notna(avg_margin) else "N/A")
            except Exception:
                st.metric("Avg Victory Margin", "N/A")
        
        # Column 3: Comeback rate (percentage of wins from grid deficit)
        with col3:
            try:
                # Calculate percentage of battles where winner overcame starting position deficit
                startDiff_data = pd.to_numeric(history_df['startDiff'], errors='coerce')
                comeback_battles = (startDiff_data > 0).sum()  # Positive startDiff = overcame deficit
                comeback_rate = comeback_battles / total_battles * 100
                st.metric("Comeback Rate", f"{comeback_rate:.1f}%")
            except Exception:
                st.metric("Comeback Rate", "N/A")
        
        # Column 4: Overall DNF (Did Not Finish) rate
        with col4:
            try:
                # Calculate DNF rate for both winners and losers
                dnf_w = pd.to_numeric(history_df['dnf_W'], errors='coerce').sum()  # Winner DNFs
                dnf_l = pd.to_numeric(history_df['dnf_L'], errors='coerce').sum()  # Loser DNFs
                total_dnfs = dnf_w + dnf_l
                # Rate = DNFs / (battles * 2 drivers per battle)
                dnf_rate = total_dnfs / (total_battles * 2) * 100
                st.metric("Overall DNF Rate", f"{dnf_rate:.1f}%")
            except Exception:
                st.metric("Overall DNF Rate", "N/A")
        
        # Analyze rating change distribution with safe handling
        st.subheader("Rating Change Distribution")
        
        try:
            # Get all valid rating changes for distribution analysis
            delta_data = pd.to_numeric(history_df['delta'], errors='coerce').dropna()
            if len(delta_data) > 0:
                # Create histogram showing distribution of Elo rating changes
                fig = px.histogram(
                    x=delta_data,
                    nbins=30,  # 30 bins for good granularity
                    title='Distribution of Rating Changes',
                    labels={'x': 'Rating Change (Elo points)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating change data available for distribution")
        except Exception as viz_error:
            st.warning(f"Could not create rating distribution chart: {viz_error}")
        
        # Analyze correlation between victory margin and rating change
        st.subheader("Victory Margin vs Rating Change")
        
        try:
            # Filter out extreme outliers for better visualization with safe handling
            margin_data = pd.to_numeric(history_df['margin'], errors='coerce')
            delta_data = pd.to_numeric(history_df['delta'], errors='coerce')
            
            # Create mask to filter valid data and remove extreme outliers
            valid_mask = (
                pd.notna(margin_data) & pd.notna(delta_data) &
                (margin_data >= 0) & (margin_data <= 10) &  # Reasonable margin range
                (delta_data <= 50)  # Reasonable rating change range
            )
            
            filtered_data = history_df[valid_mask].copy()
            
            if len(filtered_data) > 10:
                # Sample data for performance if dataset is very large
                sample_size = min(1000, len(filtered_data))
                sample_data = filtered_data.sample(sample_size) if len(filtered_data) > sample_size else filtered_data
                
                # Create scatter plot with trendline showing margin vs rating relationship
                fig = px.scatter(
                    sample_data,
                    x='margin',
                    y='delta',
                    opacity=0.6,  # Semi-transparent points to show density
                    title='Victory Margin vs Rating Gain',
                    labels={
                        'margin': 'Victory Margin (positions)',
                        'delta': 'Rating Change (Elo points)'
                    },
                    trendline='ols'  # Ordinary Least Squares trendline
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for margin vs rating change analysis")
        except Exception as viz_error:
            st.warning(f"Could not create margin vs rating chart: {viz_error}")
        
        # Analyze how racing has changed over different eras
        st.subheader("Era Analysis")
        
        try:
            # Safe aggregation by year to show historical trends
            years = pd.to_numeric(history_df['year'], errors='coerce').dropna()
            if len(years) > 0:
                era_stats = []
                
                # Calculate statistics for each year
                for year in sorted(years.unique()):
                    year_data = history_df[pd.to_numeric(history_df['year'], errors='coerce') == year]
                    
                    if len(year_data) > 0:
                        # Safe calculations for each annual metric
                        delta_mean = pd.to_numeric(year_data['delta'], errors='coerce').mean()
                        margin_mean = pd.to_numeric(year_data['margin'], errors='coerce').mean()
                        dnf_w = pd.to_numeric(year_data['dnf_W'], errors='coerce').sum()
                        dnf_l = pd.to_numeric(year_data['dnf_L'], errors='coerce').sum()
                        startDiff_data = pd.to_numeric(year_data['startDiff'], errors='coerce')
                        comeback_freq = (startDiff_data > 0).mean() if len(startDiff_data) > 0 else 0
                        
                        total_races = len(year_data)
                        # Calculate annual DNF and comeback rates
                        dnf_rate = (dnf_w + dnf_l) / (total_races * 2) * 100
                        comeback_rate = comeback_freq * 100
                        
                        era_stats.append({
                            'Year': int(year),
                            'Avg Drama Factor': delta_mean if pd.notna(delta_mean) else 0,
                            'Avg Margin': margin_mean if pd.notna(margin_mean) else 0,
                            'DNF Rate %': dnf_rate,
                            'Comeback Rate %': comeback_rate,
                            'Total Races': total_races
                        })
                
                if era_stats:
                    era_df = pd.DataFrame(era_stats)
                    safe_display(era_df, use_container_width=True)
                    
                    # Create trend visualization with error handling
                    if len(era_df) > 1:
                        try:
                            # Dual-axis chart showing drama factor and DNF rate trends over time
                            fig = go.Figure()
                            
                            # Primary y-axis: Drama Factor (average rating change)
                            fig.add_trace(go.Scatter(
                                x=era_df['Year'],
                                y=era_df['Avg Drama Factor'],
                                mode='lines+markers',
                                name='Drama Factor',
                                yaxis='y'  # Left y-axis
                            ))
                            
                            # Secondary y-axis: DNF Rate
                            fig.add_trace(go.Scatter(
                                x=era_df['Year'],
                                y=era_df['DNF Rate %'],
                                mode='lines+markers',
                                name='DNF Rate %',
                                yaxis='y2'  # Right y-axis
                            ))
                            
                            # Configure dual-axis layout
                            fig.update_layout(
                                title='Racing Trends Over Time',
                                xaxis_title='Year',
                                yaxis=dict(title='Drama Factor (Avg Rating Change)', side='left'),
                                yaxis2=dict(title='DNF Rate %', side='right', overlaying='y'),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as trend_error:
                            st.warning(f"Could not create trends visualization: {trend_error}")
                else:
                    st.info("No era analysis data available")
            else:
                st.info("No valid year data for era analysis")
        except Exception as e:
            st.error(f"Error in era analysis: {e}")
    
    except Exception as e:
        # Handle any errors in race characteristics analysis
        st.error(f"Error in race characteristics analysis: {e}")
