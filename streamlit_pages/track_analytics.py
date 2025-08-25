# Import required libraries for data processing, visualization, and UI components
import streamlit as st
import pandas as pd
import random
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go


# Step 1: Implement safe_display for safe dataframe display


def safe_display(df, **kwargs):
    """
    Safely display DataFrames in Streamlit with comprehensive error handling.
    Attempts to clean problematic data if initial display fails.
    
    This function handles common DataFrame display issues like:
    - Mixed data types in columns
    - Special characters or formatting issues
    - Index-related display problems
    
    Args:
        df: DataFrame to display
        **kwargs: Additional arguments passed to st.dataframe()
    """
    try:
        # Attempt normal DataFrame display first
        st.dataframe(df, **kwargs)
    except Exception as e:
        # If display fails, attempt to clean the DataFrame
        st.warning(f"DataFrame display error: {e}\nTrying to clean dataframe before display.")
        df_clean = df.copy()
        
        # Clean each column based on its data type
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Handle object/string columns
                try:
                    # Try to convert to numeric if possible
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception:
                    # If numeric conversion fails, clean string data
                    df_clean[col] = df_clean[col].astype(str)
                    # Remove problematic pandas indexer references that can cause display issues
                    df_clean[col] = df_clean[col].str.replace(r'_iLocIndexer|iloc|loc', '', regex=True)
        
        # Display the cleaned DataFrame
        st.dataframe(df_clean, **kwargs)



# Step 2: Implement filtering with safe default placeholder


def filter_with_placeholder(df, filter_column, selected_values, placeholder_value):
    """
    Apply filtering with intelligent fallback behavior when filters are too restrictive.
    Prevents empty results by using placeholder values or returning full dataset.
    
    This function handles common filtering scenarios:
    - Empty selection (user clears all filters)
    - Overly restrictive filters that return no results
    - Flexible fallback to placeholder data or full dataset
    
    Args:
        df: DataFrame to filter
        filter_column: Column name to apply filter on
        selected_values: List of values to filter by (empty list = no filter selected)
        placeholder_value: Fallback value/DataFrame when filtering fails
        
    Returns:
        DataFrame: Filtered data or fallback data if filtering results in empty set
    """
    if not selected_values:  # Filter cleared - handle empty selection
        if isinstance(placeholder_value, pd.DataFrame):
            # Use provided DataFrame as fallback
            return placeholder_value
        else:
            if placeholder_value is True:
                # Return full dataset if placeholder is True
                return df
            else:
                # Filter by specific placeholder value
                return df[df[filter_column] == placeholder_value]
    else:
        # Apply user-selected filter values
        mask = df[filter_column].isin(selected_values)
        filtered_df = df[mask]
        
        # Check if filtering resulted in empty dataset
        if filtered_df.empty:
            # Apply fallback logic when filter is too restrictive
            if isinstance(placeholder_value, pd.DataFrame):
                return placeholder_value
            elif placeholder_value is True:
                # Return full dataset as fallback
                return df
            else:
                # Filter by specific placeholder value as fallback
                return df[df[filter_column] == placeholder_value]
        else:
            # Return successfully filtered data
            return filtered_df



# Step 3: Main track analytics function with safe filtering


def track_analytics_page(history_df, rating_df, result_df):
    """
    Main function to render the F1 track analytics page.
    Analyzes circuit-specific performance patterns and driver dominance at different tracks.
    
    Provides two main analytical perspectives:
    - Circuit Masters: Identifies drivers who excel at specific tracks
    - Track Characteristics: Analyzes general track properties (drama, comebacks, DNFs)
    
    Args:
        history_df: DataFrame containing historical battle data with circuit information
        rating_df: DataFrame containing rating information (currently unused)
        result_df: DataFrame containing race results with final positions
    """
    # Display page header with checkered flag emoji
    st.header("🏁 Track Analytics")
    
    # Create sidebar section for track-specific filters
    st.sidebar.subheader("Track Filters")
    
    # Circuit selection dropdown - includes all unique circuits plus 'All' option
    circuits = ['All'] + sorted(history_df['circuitName'].dropna().unique().tolist())
    selected_circuit = st.sidebar.selectbox("Select Circuit", circuits)
    
    # Year selection multi-select - reverse chronological order (newest first)
    years = ['All'] + sorted(history_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect("Select Years", years, default=['All'])
    
    # Minimum races filter - ensures statistical significance for track-specific analysis
    min_races_track = st.sidebar.slider(
        "Minimum Races at Track",
        min_value=1,   # Allow analysis with minimal data
        max_value=20,  # Reasonable maximum for track-specific analysis
        value=3        # Default requires 3 races at track for meaningful statistics
    )
    
    # Apply safe filtering with comprehensive fallback handling
    filtered_df = history_df.copy()
    
    # Apply circuit filter if specific circuit selected
    if selected_circuit != 'All':
        filtered_df = filtered_df[filtered_df['circuitName'] == selected_circuit]
    
    # Apply year filter if specific years selected
    if selected_years and 'All' not in selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    
    # Safety check - warn and provide fallback if filters are too restrictive
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters. Showing default 100 most recent records.")
        # Use most recent 100 battles as fallback
        filtered_df = history_df.sort_values('date', ascending=False).head(100)
        selected_circuit = 'All'  # Reset circuit selection to show all circuits

    # Show Circuit Masters analysis only when not filtering by specific years
    # (to avoid confusion when analyzing historical dominance vs recent performance)
    if not (selected_years and 'All' not in selected_years):
        st.subheader("Circuit Masters")
        # Calculate which drivers dominate at each track
        track_dominance = calculate_track_dominance(filtered_df, min_races_track, result_df)
        # Display track dominance results
        display_track_dominance(track_dominance, selected_circuit)
    
    # Always show track characteristics analysis
    st.subheader("Track Characteristics")
    # Calculate general track properties and patterns
    track_characteristics = calculate_track_characteristics(filtered_df)
    # Display track characteristics with visualizations
    display_track_characteristics(track_characteristics)



# Step 4: Keep existing calculation and display functions (use safe_display)


def calculate_track_dominance(history_df, min_races, result_df):
    """
    Calculate driver dominance statistics for each track/circuit.
    Identifies which drivers have the best win records at specific circuits.
    
    Track dominance is measured by:
    - Win rate at the circuit (wins / total races at track)
    - Total number of wins at the track
    - Average victory margin when winning at the track
    - Minimum race threshold to ensure statistical significance
    
    Args:
        history_df: DataFrame with battle history including circuit information
        min_races: Minimum number of races at track required for inclusion
        result_df: DataFrame with race results to identify actual race winners
        
    Returns:
        dict: Nested dictionary with track names as keys and sorted driver statistics as values
    """
    track_stats = {}

    # Merge battle history with race results to identify actual race winners
    result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
    result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

    # Join battle data with race results
    history_df = pd.merge(history_df, result_filtered_df, on=["year", "round", "winnerId"])
    # Note: Not filtering by finalPosition==1 here to include all battles, 
    # but we'll filter for actual wins in the win calculation below
    
    # Process each unique circuit/track
    for track in history_df['circuitName'].unique():
        # Get all battles at this specific track
        track_data = history_df[history_df['circuitName'] == track]
        driver_stats = {}
        
        # Analyze each driver's performance at this track
        for driver in pd.concat([track_data['winnerName'], track_data['loserName']]).unique():
            # Get all races where this driver participated at this track
            driver_races = track_data[
                (track_data['winnerName'] == driver) | (track_data['loserName'] == driver)
            ]
            
            # Only include drivers with sufficient races at this track for statistical significance
            if len(driver_races) >= min_races:
                # Count actual race wins (finalPosition == 1) where driver was battle winner
                wins = len(track_data[(track_data['winnerName'] == driver) & (track_data['finalPosition'] == 1)])
                # Calculate win rate at this track
                win_rate = wins / len(driver_races) if len(driver_races) > 0 else 0
                # Calculate average victory margin for wins at this track
                avg_margin = track_data[(track_data['winnerName'] == driver) & (track_data['finalPosition'] == 1)]['margin'].mean()
                
                # Store comprehensive track performance statistics for this driver
                driver_stats[driver] = {
                    'races': len(driver_races),                    # Total races at this track
                    'wins': wins,                                  # Actual race wins at this track
                    'win_rate': win_rate,                         # Win percentage at this track
                    'avg_margin': avg_margin if not pd.isna(avg_margin) else 0  # Average victory margin
                }
        
        # Sort drivers by track performance (win rate first, then total races as tiebreaker)
        sorted_drivers = sorted(
            driver_stats.items(),
            key=lambda x: (x[1]['win_rate'], x[1]['races']),  # Sort by win rate, then by experience
            reverse=True  # Best performers first
        )
        
        # Store sorted driver performance data for this track
        track_stats[track] = sorted_drivers
    
    return track_stats



def display_track_dominance(track_dominance, selected_circuit):
    """
    Display track dominance results with appropriate formatting and visualizations.
    Shows either specific circuit analysis or summary across all circuits.
    
    Args:
        track_dominance: Dictionary containing track dominance statistics from calculate_track_dominance()
        selected_circuit: Either specific circuit name or 'All' for summary view
    """
    # print(track_dominance)  # Debug print (commented out)
    
    if selected_circuit != 'All':
        # Display detailed analysis for specific circuit
        if selected_circuit in track_dominance:
            dominance_data = track_dominance[selected_circuit]
            
            st.write(f"**Top Performers at {selected_circuit}:**")
            # Create formatted table of top performers at this circuit
            dominance_df = pd.DataFrame([
                {
                    'Driver': driver,
                    'Races': stats['races'],                                    # Total races at track
                    'Wins': stats['wins'],                                      # Race wins at track
                    'Win Rate %': float(f"{stats['win_rate']*100:.1f}"),      # Win percentage formatted
                    'Avg Margin': float(f"{stats['avg_margin']:.2f}")          # Average victory margin formatted
                }
                for driver, stats in dominance_data[:10]  # Show top 10 performers
            ])
            safe_display(dominance_df, use_container_width=True)
            
            # Create bar chart visualization of win rates if we have data
            if len(dominance_data) > 0:
                # Prepare data for visualization (limit driver name length for readability)
                viz_data = pd.DataFrame([
                    {'Driver': driver[:20], 'Win Rate': stats['win_rate']*100}  # Truncate long names
                    for driver, stats in dominance_data[:10]  # Top 10 only for chart clarity
                ])
                # Create bar chart showing win rates at this circuit
                fig = px.bar(
                    viz_data,
                    x='Driver',
                    y='Win Rate',
                    title=f'Win Rates at {selected_circuit}'
                )
                fig.update_xaxes(tickangle=45)  # Rotate driver names for readability
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Display summary view across all circuits
        summary_data = []
        # Create summary showing best driver at each track
        for track, drivers in track_dominance.items():
            if drivers:  # Only include tracks with qualifying drivers
                best_driver, best_stats = drivers[0]  # Get top performer at this track
                summary_data.append({
                    'Track': track,
                    'Best Driver': best_driver,
                    'Wins': best_stats['wins'],
                    'Win Rate': f"{best_stats['win_rate']*100:.1f}%",  # Format as percentage string
                    'Races': best_stats['races']
                })
        
        # Sort summary by win rate and display
        summary_df = pd.DataFrame(summary_data).sort_values('Win Rate', ascending=False)
        safe_display(summary_df, use_container_width=True)



def calculate_track_characteristics(history_df):
    """
    Calculate general characteristics and patterns for each track/circuit.
    Analyzes track properties that affect racing patterns and excitement levels.
    
    Track characteristics measured:
    - Drama factor: Average Elo rating change (higher = more unpredictable/exciting racing)
    - Comeback frequency: Percentage of races with significant position gains
    - DNF rate: Mechanical failure and incident rate at the track
    - Average victory margin: How close battles typically are at this track
    
    Args:
        history_df: DataFrame containing battle history with track and performance data
        
    Returns:
        dict: Dictionary with track names as keys and characteristic statistics as values
    """
    characteristics = {}
    
    # Process each unique circuit/track
    for track in history_df['circuitName'].unique():
        # Get all battle data for this specific track
        track_data = history_df[history_df['circuitName'] == track]
        
        # Calculate drama factor - average Elo rating change magnitude
        # Higher values indicate more unpredictable/exciting racing
        drama_factor = track_data['delta'].mean()
        
        # Calculate comeback frequency - percentage of battles with significant position gains
        # Battles where winner overcame 3+ grid position deficit
        comeback_rate = len(track_data[track_data['startDiff'] > 3]) / len(track_data)
        
        # Calculate DNF (Did Not Finish) rate - mechanical reliability at this track
        # Combined DNF rate for both battle participants
        dnf_rate = (track_data['dnf_W'].sum() + track_data['dnf_L'].sum()) / (len(track_data) * 2)
        
        # Calculate average victory margin - how close battles are at this track
        avg_margin = track_data['margin'].mean()
        
        # Store comprehensive track characteristics
        characteristics[track] = {
            'drama_factor': drama_factor,           # Average rating change (excitement measure)
            'comeback_frequency': comeback_rate,    # Percentage of comeback victories
            'dnf_rate': dnf_rate,                  # Mechanical failure rate
            'avg_margin': avg_margin,              # Average victory margin
            'total_races': len(track_data)         # Total battles analyzed
        }
    
    return characteristics



def display_track_characteristics(characteristics):
    """
    Display track characteristics with formatted tables and visualizations.
    Shows racing patterns and properties that make each circuit unique.
    
    Creates two main visualizations:
    - Bar chart of most dramatic tracks (by rating changes)
    - Scatter plot showing relationship between comeback rate and DNF rate
    
    Args:
        characteristics: Dictionary containing track characteristics from calculate_track_characteristics()
    """
    # Prepare formatted data for display table
    char_data = []
    for track, stats in characteristics.items():
        char_data.append({
            'Track': track,
            'Drama Factor': float(f"{stats['drama_factor']:.1f}"),        # Average rating change
            'Comeback %': float(f"{stats['comeback_frequency']*100:.1f}"), # Comeback percentage
            'DNF Rate %': float(f"{stats['dnf_rate']*100:.1f}"),          # DNF percentage
            'Avg Margin': float(f"{stats['avg_margin']:.2f}"),            # Average victory margin
            'Total Battles': float(stats['total_races'])                   # Sample size
        })
    
    # Create DataFrame and sort by drama factor (most exciting tracks first)
    char_df = pd.DataFrame(char_data)
    char_df_sorted = char_df.sort_values('Drama Factor', ascending=False)
    
    # Display comprehensive track characteristics table
    safe_display(char_df_sorted, use_container_width=True)
    
    # Create two-column layout for visualizations
    col1, col2 = st.columns(2)
    
    # Column 1: Most dramatic tracks bar chart
    with col1:
        # Bar chart showing tracks with highest drama factors (rating changes)
        fig1 = px.bar(
            char_df_sorted.head(10),  # Top 10 most dramatic tracks
            x='Track', 
            y='Drama Factor', 
            title='Most Dramatic Tracks (by Rating Changes)'
        )
        fig1.update_xaxes(tickangle=45)  # Rotate track names for readability
        st.plotly_chart(fig1, use_container_width=True)
    
    # Column 2: Comeback vs DNF rate scatter plot
    with col2:
        # Scatter plot showing relationship between comeback opportunities and reliability
        fig2 = px.scatter(
            char_df, 
            x='Comeback %',     # X-axis: Comeback frequency
            y='DNF Rate %',     # Y-axis: DNF rate
            size='Total Battles',  # Bubble size by sample size
            hover_name='Track',    # Track name on hover
            title='Comeback Rate vs DNF Rate'
        )
        st.plotly_chart(fig2, use_container_width=True)



# def create_track_specialization_heatmap(history_df, min_races, result_df):
#     driver_stats = defaultdict(lambda: {'races': 0, 'rating': 0, 'rating_sum': 0, 'rating_latest':1500})

#     result_filtered_df = result_df[["year", "round", "driverId", "finalPosition"]]
#     result_filtered_df = result_filtered_df.rename(columns={"driverId":"winnerId"})

#     history_df = pd.merge(history_df, result_filtered_df, on=["year", "round", "winnerId"])
#     # history_df = history_df[history_df["finalPosition"]==1]

#     for _, row in history_df.iterrows():
#         driver_stats[row['winnerName']]['races'] += 1
#         driver_stats[row['winnerName']]['rating_latest'] = row['Rw_new']
#         driver_stats[row['winnerName']]['rating_sum'] += row['Rw_new']
#         driver_stats[row['winnerName']]['rating'] = driver_stats[row['winnerName']]['rating_sum']/driver_stats[row['winnerName']]['races']

#         driver_stats[row['loserName']]['races'] += 1
#         driver_stats[row['loserName']]['rating_latest'] = row['Rl_new']
#         driver_stats[row['loserName']]['rating_sum'] += row['Rl_new']
#         driver_stats[row['loserName']]['rating'] = driver_stats[row['loserName']]['rating_sum']/driver_stats[row['loserName']]['races']
#     top_drivers = sorted(
#         [(driver, stats) for driver, stats in driver_stats.items() if stats['races'] >= 10],
#         key=lambda x: x[1]['rating_latest'],
#         reverse=True
#     )[:15]
#     top_driver_names = [driver for driver, _ in top_drivers]
#     heatmap_data = []
#     tracks = sorted(history_df['circuitName'].unique())
#     for driver in top_driver_names:
#         driver_row = []
#         for track in tracks:
#             track_driver_data = history_df[
#                 (history_df['circuitName'] == track) &
#                 ((history_df['winnerName'] == driver) | (history_df['loserName'] == driver))
#             ]
#             if len(track_driver_data) >= min_races:
#                 wins = len(track_driver_data[(track_driver_data['winnerName'] == driver) & (track_driver_data['finalPosition'] == 1)])
#                 win_rate = wins / len(track_driver_data)
#                 driver_row.append(win_rate)
#             else:
#                 driver_row.append(None)
#         heatmap_data.append(driver_row)
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data,
#         x=[track[:15] for track in tracks],
#         y=[driver[:15] for driver in top_driver_names],
#         colorscale='RdYlBu_r',
#         zmin=0,
#         zmax=1
#     ))
#     fig.update_layout(title='Driver-Track Specialization Heatmap (Win Rates)', xaxis_title='Tracks', yaxis_title='Top Drivers', height=600)
#     st.plotly_chart(fig, use_container_width=True)