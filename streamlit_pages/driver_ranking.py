# Import required libraries for data processing, visualization, and UI
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict

# Import utility functions from local util module
from util import *


def driver_rankings_page(history_df, rating_df):
    """
    Main function to render the F1 driver rankings and ratings page.
    Displays current Elo rankings, rating evolution charts, and driver statistics.
    
    Args:
        history_df: DataFrame containing historical race battle data
        rating_df: DataFrame containing rating information (currently unused in function)
    """
    # Display page header with trophy emoji
    st.header("🏆 Driver Rankings & Ratings")
    
    # **STEP 1: Safe data preprocessing**
    # Create a copy to avoid modifying the original DataFrame
    history_df = history_df.copy()
    # Convert date column to datetime format, coercing errors to NaT (Not a Time)
    history_df['date'] = pd.to_datetime(history_df['date'], errors='coerce')
    
    # Create sidebar section for user filters
    st.sidebar.subheader("Filters")
    
    # **STEP 2: Safe date filter with bounds checking**
    # Attempt to get date range from data with comprehensive error handling
    try:
        # Extract minimum and maximum dates from the dataset
        min_date = history_df['date'].min()
        max_date = history_df['date'].max()
        
        # Check if dates are valid (not NaT - Not a Time)
        if pd.isna(min_date) or pd.isna(max_date):
            # Display warning and set fallback date range
            st.warning("⚠️ Invalid date data detected. Using fallback date range.")
            min_date = pd.Timestamp('2020-01-01')
            max_date = pd.Timestamp.now()
    except Exception:
        # If any exception occurs, use safe fallback dates
        min_date = pd.Timestamp('2020-01-01')
        max_date = pd.Timestamp.now()
    
    # Create date range selector in sidebar with calculated bounds
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date.date(), max_date.date()),  # Default to full range
        min_value=min_date.date(),  # Set minimum selectable date
        max_value=max_date.date(),  # Set maximum selectable date
        help="Clear to use smart defaults"  # Tooltip text
    )
    
    # **STEP 3: Safe GP filter**
    # Determine maximum round number with error handling
    try:
        # Get the highest round number from the data
        max_rounds = int(history_df['round'].max())
        # Validate that we have a sensible round number
        if pd.isna(max_rounds) or max_rounds < 1:
            max_rounds = 23  # Typical F1 season has ~23 races
    except Exception:
        # Default to typical F1 season length if error occurs
        max_rounds = 23
    
    # Create slider for minimum round filter (ratings calculated from this round onwards)
    min_round_filter = st.sidebar.slider(
        "Rating after Round X onwards",
        min_value=1,
        max_value=max_rounds,
        value=1  # Default to include all rounds
    )
    
    # **STEP 4: Safe constructor filter**
    # Get unique constructors and add 'All' option
    constructors = ['All'] + sorted(history_df['constructor'].dropna().unique().tolist())
    # Create dropdown selector for constructor filtering
    selected_constructor = st.sidebar.selectbox("Constructor", constructors)
    
    # Create slider for minimum races requirement filter
    min_races = st.sidebar.slider(
        "Minimum Races",
        min_value=1,
        max_value=50,
        value=20  # Default requires 20 races for statistical significance
    )
    
    # **STEP 5: Apply safe filtering with fallbacks**
    # Apply all filters using the safe filtering function
    filtered_history, used_fallback = safe_filter_driver_rankings(
        history_df, date_range, min_round_filter, selected_constructor, min_races
    )
    
    # Display current data status in expandable section
    with st.expander("📈 Current Data Summary"):
        # Create four columns for metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total number of head-to-head battles
        with col1:
            st.metric("Total Battles", len(filtered_history))
        
        # Column 2: Count unique drivers (from both winner and loser columns)
        with col2:
            st.metric("Unique Drivers", pd.concat([filtered_history['winnerName'], filtered_history['loserName']]).nunique())
        
        # Column 3: Display actual date range of filtered data
        with col3:
            # Format date range string safely
            date_range_str = f"{filtered_history['date'].min().strftime('%Y-%m-%d')} to {filtered_history['date'].max().strftime('%Y-%m-%d')}" if not filtered_history.empty else "N/A"
            st.write(f"**Date Range:** {date_range_str}")
        
        # Column 4: Show selected constructor filter
        with col4:
            st.metric("Constructor", selected_constructor if selected_constructor != 'All' else "All Teams")
    
    # **STEP 7: Calculate current ratings with safety checks**
    try:
        # Calculate current Elo ratings from filtered battle history
        current_ratings = calculate_current_ratings(filtered_history, min_races)
        
        # If no drivers meet minimum races requirement, reduce the requirement
        if current_ratings.empty:
            st.warning("⚠️ No drivers meet the minimum races requirement. Reducing requirement...")
            # Try with half the minimum requirement
            current_ratings = calculate_current_ratings(filtered_history, max(1, min_races // 2))
        
    except Exception as e:
        # Handle any errors in rating calculation
        st.error(f"Error calculating ratings: {e}")
        st.info("Using fallback data...")
        # Use last 100 races as fallback data
        fallback_data = history_df.tail(100)
        current_ratings = calculate_current_ratings(fallback_data, 1)
    
    # If still no data available, show error and exit
    if current_ratings.empty:
        st.error("❌ No rating data available. Please check your data or adjust filters.")
        return
    
    # **STEP 8: Display current rankings (unchanged)**
    # Create two-column layout for rankings table and distribution chart
    col1, col2 = st.columns([2, 1])
    
    # Left column: Current Elo rankings table
    with col1:
        st.subheader("Current Elo Rankings")
        # Display formatted rankings table
        display_rankings_table(current_ratings)
    
    # Right column: Rating distribution histogram
    with col2:
        st.subheader("Rating Distribution")
        if len(current_ratings) > 0:
            # Create histogram of rating distribution
            fig = px.histogram(
                current_ratings, 
                x='rating', 
                nbins=min(20, len(current_ratings)),  # Limit bins to prevent overcrowding
                title="Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for distribution chart")
    
    # Rating evolution charts section
    st.subheader("Rating Evolution Over Time")
    
    # Only show timeline if we have drivers to display
    if len(current_ratings) > 0:
        # Get list of available drivers from current ratings
        available_drivers = current_ratings['driverName'].tolist()
        # Default to showing top 5 drivers
        default_drivers = current_ratings['driverName'].head(min(5, len(current_ratings))).tolist()
        
        # Multi-select widget for choosing drivers to display in timeline
        selected_drivers = st.multiselect(
            "Select Drivers for Timeline",
            options=available_drivers,
            default=default_drivers
        )
        
        # Create and display timeline chart if drivers are selected
        if selected_drivers:
            rating_timeline_chart(filtered_history, selected_drivers)
    else:
        st.info("No drivers available for timeline chart")
    
    # **STEP 9: Top performers metrics with safe handling**
    # Display key statistics in four-column layout
    if len(current_ratings) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Highest rated driver
        with col1:
            try:
                # Get top-rated driver (first row after sorting by rating desc)
                highest_rated = current_ratings.iloc[0]
                st.metric(
                    "Highest Rated Driver",
                    f"{highest_rated['driverName']}",
                    f"{highest_rated['rating']:.0f} Elo"
                )
            except Exception:
                # Fallback if error occurs
                st.metric("Highest Rated Driver", "N/A", "No data")
        
        # Column 2: Most improved driver
        with col2:
            try:
                # Calculate driver with biggest rating improvement
                most_improved = get_most_improved_driver(filtered_history)
                st.metric(
                    "Most Improved",
                    # Truncate long names to fit display
                    most_improved['driver'][:15] if most_improved['driver'] != 'N/A' else 'N/A',
                    f"+{most_improved['improvement']:.0f} Elo" if most_improved['improvement'] > 0 else "No data"
                )
            except Exception:
                st.metric("Most Improved", "N/A", "No data")
        
        # Column 3: Most active driver (highest number of races)
        with col3:
            try:
                # Find driver with most races
                most_active = current_ratings.loc[current_ratings['matches'].idxmax()]
                st.metric(
                    "Most Active",
                    most_active['driverName'],
                    f"{most_active['matches']} races"
                )
            except Exception:
                st.metric("Most Active", "N/A", "No data")
        
        # Column 4: Average rating across all qualified drivers
        with col4:
            try:
                # Calculate mean rating
                avg_rating = current_ratings['rating'].mean()
                st.metric(
                    "Average Rating",
                    f"{avg_rating:.0f}",
                    "Elo"
                )
            except Exception:
                st.metric("Average Rating", "N/A", "No data")


def safe_filter_driver_rankings(history_df, date_range, min_round_filter, selected_constructor, min_races):
    """
    Apply safe filtering with smart defaults and fallbacks to prevent empty datasets.
    Uses progressive fallback strategy when filters are too restrictive.
    
    Args:
        history_df: DataFrame with historical race data
        date_range: Tuple of (start_date, end_date) for filtering
        min_round_filter: Minimum round number to include
        selected_constructor: Constructor to filter by ('All' for no filter)
        min_races: Minimum number of races (used for info, not filtering here)
    
    Returns:
        tuple: (filtered_history_df, used_fallback_boolean)
    """
    # Start with a copy of the original data
    filtered_history = history_df.copy()
    used_fallback = False  # Track if we had to use fallback data
    
    try:
        # **Safe date filtering**
        # Only apply date filter if valid date range provided
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            # Filter data within the specified date range
            date_filtered = filtered_history[
                (filtered_history['date'] >= pd.to_datetime(start_date)) &
                (filtered_history['date'] <= pd.to_datetime(end_date))
            ]
            
            # Check if date filtering resulted in empty dataset
            if date_filtered.empty:
                # Fallback to recent 2 years of data
                recent_cutoff = history_df['date'].max() - pd.Timedelta(days=730)
                filtered_history = filtered_history[filtered_history['date'] >= recent_cutoff]
                used_fallback = True
            else:
                # Use the date-filtered data
                filtered_history = date_filtered
        
        # **Safe round filtering**
        # Filter by minimum round number
        round_filtered = filtered_history[filtered_history['round'] >= min_round_filter]
        if round_filtered.empty:
            # Keep original data if round filtering removes everything
            st.warning(f"Round filter >= {min_round_filter} resulted in no data. Using all rounds.")
        else:
            # Apply the round filter
            filtered_history = round_filtered
        
        # **Safe constructor filtering**
        # Only filter if specific constructor selected
        if selected_constructor != 'All':
            constructor_filtered = filtered_history[
                filtered_history['constructor'] == selected_constructor
            ]
            
            # Check if constructor filtering resulted in empty dataset
            if constructor_filtered.empty:
                st.warning(f"No data found for {selected_constructor}. Showing all constructors.")
            else:
                # Apply constructor filter
                filtered_history = constructor_filtered
        
        # **Final safety check**
        # Ensure we have some data after all filtering
        if filtered_history.empty:
            st.warning("⚠️ Filters resulted in no data. Using recent 100 races as fallback.")
            # Use most recent 100 races as absolute fallback
            filtered_history = history_df.sort_values('date', ascending=False).head(100)
            used_fallback = True
        
        # **Ensure minimum data for meaningful analysis**
        # Expand dataset if we have too few data points
        if len(filtered_history) < 10:
            st.info("📊 Limited data with current filters. Expanding to show more results...")
            # Get more recent data for better analysis
            recent_data = history_df.sort_values('date', ascending=False).head(200)
            if len(recent_data) > len(filtered_history):
                filtered_history = recent_data
                used_fallback = True
    
    except Exception as e:
        # Handle any unexpected errors in filtering process
        st.error(f"Error in filtering: {e}")
        st.info("Using fallback data...")
        # Use last 100 races as emergency fallback
        filtered_history = history_df.tail(100)
        used_fallback = True
    
    return filtered_history, used_fallback


def calculate_current_ratings(history_df, min_races):
    """
    Calculate current Elo ratings from historical battle data.
    Processes battles chronologically to get final rating for each driver.
    
    Args:
        history_df: DataFrame containing race battle history
        min_races: Minimum number of races required for a driver to be included
        
    Returns:
        DataFrame: Current ratings sorted by rating (highest first)
    """
    # Dictionary to store final ratings for each driver
    current_ratings = {}
    # Counter for number of races per driver
    matches_count = defaultdict(int)
    
    # Return empty DataFrame if no input data
    if history_df.empty:
        return pd.DataFrame()
    
    try:
        # Process battles chronologically to track rating evolution
        for _, row in history_df.sort_values('date').iterrows():
            # Get driver identifiers (prefer ID over name for consistency)
            winner_id = row['winnerId'] if 'winnerId' in row else row['winnerName']
            loser_id = row['loserId'] if 'loserId' in row else row['loserName']
            # Get driver display names
            winner_name = row['winnerName']
            loser_name = row['loserName']
            
            # Safe conversion of rating values to numeric
            winner_rating = pd.to_numeric(row['Rw_new'], errors='coerce')
            loser_rating = pd.to_numeric(row['Rl_new'], errors='coerce')
            
            # Only process if both ratings are valid numbers
            if pd.notna(winner_rating) and pd.notna(loser_rating):
                # Update current ratings (overwrites previous, giving us final rating)
                current_ratings[winner_id] = {
                    'rating': float(winner_rating),
                    'driverName': winner_name
                }
                current_ratings[loser_id] = {
                    'rating': float(loser_rating), 
                    'driverName': loser_name
                }
                
                # Increment race counter for both drivers
                matches_count[winner_id] += 1
                matches_count[loser_id] += 1
        
        # Convert to DataFrame format for display
        ratings_data = []
        for driver_id, data in current_ratings.items():
            # Only include drivers who meet minimum race requirement
            if matches_count[driver_id] >= min_races:
                ratings_data.append({
                    'driverId': driver_id,
                    'driverName': data['driverName'],
                    'rating': data['rating'],
                    'matches': matches_count[driver_id],
                    # Confidence score based on number of races (maxes at 20 races)
                    'confidence': min(1.0, matches_count[driver_id] / 20)
                })
        
        # Return DataFrame sorted by rating (highest first)
        if len(ratings_data) > 0:
            return pd.DataFrame(ratings_data).sort_values('rating', ascending=False)
        
        return pd.DataFrame(ratings_data)
    
    except Exception as e:
        # Handle any errors in rating calculation
        st.error(f"Error calculating current ratings: {e}")
        return pd.DataFrame()


def display_rankings_table(ratings_df):
    """
    Display driver rankings in a formatted table with proper column headers.
    
    Args:
        ratings_df: DataFrame containing rating data for drivers
    """
    # Handle empty data case
    if ratings_df.empty:
        st.info("No rankings data to display")
        return
    
    try:
        # Create display-friendly DataFrame
        display_df = ratings_df.copy()
        # Add rank column (1, 2, 3, ...)
        display_df['Rank'] = range(1, len(display_df) + 1)
        # Round ratings to whole numbers and handle NaN values
        display_df['Rating'] = pd.to_numeric(display_df['rating'], errors='coerce').round(0).astype('Int64')
        # Convert confidence to percentage
        display_df['Confidence'] = (pd.to_numeric(display_df['confidence'], errors='coerce') * 100).round(1)
        
        # Display table with renamed columns for better presentation
        safe_display(
            display_df[['Rank', 'driverName', 'Rating', 'matches', 'Confidence']].rename(columns={
                'driverName': 'Driver',
                'matches': 'Races',
                'Confidence': 'Confidence %'
            }),
            use_container_width=True
        )
    except Exception as e:
        # Handle display errors gracefully
        st.error(f"Error displaying rankings table: {e}")


def rating_timeline_chart(history_df, selected_drivers):
    """
    Create an interactive line chart showing rating evolution over time for selected drivers.
    
    Args:
        history_df: DataFrame containing historical race battle data
        selected_drivers: List of driver names to include in timeline
    """
    # Handle empty data or no selected drivers
    if history_df.empty or not selected_drivers:
        st.info("No data available for timeline chart")
        return
    
    try:
        # List to store timeline data points
        timeline_data = []
        
        # Process each selected driver
        for driver in selected_drivers:
            # Get all races where this driver participated (as winner or loser)
            driver_history = history_df[
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            ].sort_values('date')  # Sort chronologically
            
            # Extract rating data for each race
            for _, race in driver_history.iterrows():
                try:
                    # Determine if driver won or lost this battle
                    if race['winnerName'] == driver:
                        # Get winner's new rating
                        rating = pd.to_numeric(race['Rw_new'], errors='coerce')
                    else:
                        # Get loser's new rating
                        rating = pd.to_numeric(race['Rl_new'], errors='coerce')
                    
                    # Only add valid rating data points
                    if pd.notna(rating):
                        timeline_data.append({
                            'Driver': driver,
                            'Date': pd.to_datetime(race['date']),
                            'Rating': float(rating),
                            'Race': race.get('raceName', 'Unknown Race')  # Race name for hover info
                        })
                except Exception:
                    # Skip invalid data points
                    continue
        
        # Create and display line chart if we have data
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            # Create interactive line chart with Plotly
            fig = px.line(
                timeline_df,
                x='Date',
                y='Rating',
                color='Driver',  # Different color for each driver
                title='Driver Rating Evolution',
                hover_data=['Race']  # Show race name on hover
            )
            # Set chart height
            fig.update_layout(height=500)
            # Display chart with full container width
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available for selected drivers")
    
    except Exception as e:
        # Handle chart creation errors
        st.error(f"Error creating timeline chart: {e}")


def get_most_improved_driver(history_df):
    """
    Find the driver with the biggest rating improvement over their career.
    Compares first race rating to most recent race rating.
    
    Args:
        history_df: DataFrame containing historical race battle data
        
    Returns:
        dict: Contains 'driver' name and 'improvement' value
    """
    # Handle empty or None input data
    if history_df is None or len(history_df) == 0:
        return {'driver': 'N/A', 'improvement': 0}
    
    # Dictionary to store improvement for each driver
    driver_improvements = {}
    
    try:
        # Get all unique drivers from winner and loser columns
        all_drivers = pd.concat([
            history_df['winnerName'], 
            history_df['loserName']
        ]).dropna().unique()
        
        # Process each driver
        for driver in all_drivers:
            # Skip invalid driver names
            if pd.isna(driver) or driver == '':
                continue
                
            # Get all races for this driver (as winner or loser)
            driver_mask = (
                (history_df['winnerName'] == driver) | 
                (history_df['loserName'] == driver)
            )
            driver_races = history_df[driver_mask].sort_values('date').copy()
            
            # Need at least 3 races for meaningful improvement calculation
            if len(driver_races) < 3:
                continue
            
            # Get rating from first race
            first_race_row = driver_races.iloc[0]
            try:
                # Determine if driver won or lost first race and get appropriate rating
                if first_race_row['winnerName'] == driver:
                    first_rating = pd.to_numeric(first_race_row['Rw_new'], errors='coerce')
                else:
                    first_rating = pd.to_numeric(first_race_row['Rl_new'], errors='coerce')
                
                # Skip if rating is invalid
                if pd.isna(first_rating):
                    continue
                first_rating = float(first_rating)
                    
            except Exception:
                continue
            
            # Get rating from last race
            last_race_row = driver_races.iloc[-1]
            try:
                # Determine if driver won or lost last race and get appropriate rating
                if last_race_row['winnerName'] == driver:
                    last_rating = pd.to_numeric(last_race_row['Rw_new'], errors='coerce')
                else:
                    last_rating = pd.to_numeric(last_race_row['Rl_new'], errors='coerce')
                
                # Skip if rating is invalid
                if pd.isna(last_rating):
                    continue
                last_rating = float(last_rating)
                    
            except Exception:
                continue
            
            # Calculate improvement (can be negative if driver declined)
            improvement = last_rating - first_rating
            driver_improvements[driver] = improvement
    
    except Exception as e:
        # Return error indicator if processing fails
        return {'driver': 'Error', 'improvement': 0}
    
    # Handle case where no valid improvements were calculated
    if not driver_improvements:
        return {'driver': 'N/A', 'improvement': 0}
    
    # Find driver with maximum improvement
    best_driver = max(driver_improvements, key=driver_improvements.get)
    return {
        'driver': best_driver,
        'improvement': driver_improvements[best_driver]
    }
