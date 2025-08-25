# Import required libraries for data processing, visualization, and UI
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go

# Import utility functions from local util module
from util import *


def head_to_head_page(history_df, rating_df):
    """
    Main function to render the F1 head-to-head analysis page.
    Analyzes teammate battles between drivers, showing direct comparisons
    and overall teammate battle statistics.
    
    Args:
        history_df: DataFrame containing historical teammate battle data
        rating_df: DataFrame containing rating information (currently unused)
    """
    # Display page header with crossed swords emoji
    st.header("⚔️ Head-to-Head Analysis")
    
    # **STEP 1: Add sidebar filters for safe filtering**
    # Create sidebar section for filtering options
    st.sidebar.subheader("Head-to-Head Filters")
    
    # Year filter - multi-select with reverse chronological order (newest first)
    years = ['All'] + sorted(history_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Select Years", 
        years, 
        default=['All'],  # Default to all years
        help="Clear all = recent 5 years"  # Tooltip explaining fallback behavior
    )
    
    # Constructor/team filter - multi-select with all teams available
    constructors = ['All'] + sorted(history_df['constructor'].dropna().unique().tolist())
    selected_constructors = st.sidebar.multiselect(
        "Select Teams", 
        constructors, 
        default=['All'],  # Default to all teams
        help="Clear all = all teams"  # Tooltip explaining fallback behavior
    )
    
    # Minimum battles filter - slider to ensure statistical significance
    min_battles = st.sidebar.slider(
        "Minimum Battles Between Drivers",
        min_value=1,
        max_value=20,
        value=3  # Default requires at least 3 battles for meaningful comparison
    )
    
    # **STEP 2: Apply safe filtering**
    # Apply filters using safe filtering function with fallback handling
    filtered_df, used_defaults = safe_filter_head_to_head(
        history_df, selected_years, selected_constructors
    )
    
    # Display current data status in expandable section
    with st.expander("📈 Current Data Summary"):
        # Create four columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Total number of teammate battles
        with col1:
            st.metric("Total Battles", len(filtered_df))
        
        # Column 2: Count unique drivers involved in battles
        with col2:
            st.metric("Unique Drivers", pd.concat([filtered_df['winnerName'], filtered_df['loserName']]).nunique())
        
        # Column 3: Year range of filtered data
        with col3:
            st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if not filtered_df.empty else "N/A")
        
        # Column 4: Number of different teams/constructors
        with col4:
            st.metric("Teams", filtered_df['constructor'].nunique() if not filtered_df.empty else 0)
    
    # **STEP 3: Safe driver selection**
    # Driver selection section with comprehensive error handling
    try:
        # Get list of all available drivers from filtered data
        all_drivers = safe_get_driver_list(filtered_df)
        
        # Check if we have drivers available for comparison
        if not all_drivers:
            st.error("❌ No drivers available with current filters. Please adjust your selection.")
            return
        
        # Create two-column layout for driver selection
        col1, col2 = st.columns(2)
        # Column 1: First driver selector
        with col1:
            driver1 = st.selectbox("Select Driver 1", all_drivers, key="driver1")
        # Column 2: Second driver selector  
        with col2:
            driver2 = st.selectbox("Select Driver 2", all_drivers, key="driver2")
        
        # Only proceed with analysis if two different drivers selected
        if driver1 != driver2:
            st.subheader(f"{driver1} vs {driver2} - Teammate Record")
            # Perform detailed head-to-head analysis
            head_to_head_analysis(filtered_df, driver1, driver2)
        else:
            # Show helpful message when same driver selected twice
            st.info("👆 Please select two different drivers to compare")
    
    except Exception as e:
        # Handle any errors in driver selection process
        st.error(f"Error in driver selection: {e}")
        return
    
    # **STEP 4: Overall analysis with safe handling**
    # Show comprehensive teammate battle statistics
    st.subheader("All-Time Teammate Battle Records")
    # Display rankings of all teammate pairings
    teammate_battle_rankings(filtered_df, min_battles)
    
    st.subheader("Most Competitive Teammate Pairings")
    # Show the most evenly matched teammate pairs
    competitive_pairings(filtered_df, min_battles)


def safe_filter_head_to_head(history_df, selected_years, selected_constructors):
    """
    Apply safe filtering with smart defaults and fallbacks to prevent empty datasets.
    Uses progressive fallback strategy when user selections are too restrictive.
    
    Args:
        history_df: DataFrame with historical teammate battle data
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
            # Default: use recent 5 years for manageable dataset size
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
        
        # Final safety check - ensure we have some data
        if filtered_df.empty:
            st.warning("⚠️ Filters resulted in no data. Using recent 100 battles as fallback.")
            # Use most recent 100 battles as absolute fallback
            filtered_df = history_df.sort_values('date', ascending=False).head(100)
            used_defaults = True
        
        # Ensure minimum data for meaningful analysis
        # Expand dataset if we have too few data points for statistical relevance
        if len(filtered_df) < 20:
            st.info("📊 Limited data with current filters. Expanding to show more results...")
            # Get more recent data for better analysis
            recent_data = history_df.sort_values('date', ascending=False).head(500)
            if len(recent_data) > len(filtered_df):
                filtered_df = recent_data
                used_defaults = True
    
    except Exception as e:
        # Handle any unexpected errors in filtering process
        st.error(f"Error in filtering: {e}")
        # Use last 200 battles as emergency fallback
        filtered_df = history_df.tail(200)
        used_defaults = True
    
    return filtered_df, used_defaults


def safe_get_driver_list(history_df):
    """
    Safely extract unique driver names from history data.
    Handles missing data and invalid entries gracefully.
    
    Args:
        history_df: DataFrame containing battle history with winner/loser names
        
    Returns:
        list: Sorted list of unique driver names
    """
    try:
        # Handle empty DataFrame
        if history_df.empty:
            return []
        
        # Combine winner and loser names to get all drivers
        all_drivers = pd.concat([
            history_df['winnerName'], 
            history_df['loserName']
        ]).dropna().unique()  # Remove NaN values and get unique names
        
        # Filter out invalid entries and sort alphabetically
        return sorted([driver for driver in all_drivers if pd.notna(driver) and driver.strip()])
    
    except Exception as e:
        # Handle any errors in driver extraction
        st.warning(f"Error getting driver list: {e}")
        return []


# **KEEP ALL ANALYSIS FUNCTIONS WITH SAFE HANDLING**


def head_to_head_analysis(history_df, driver1, driver2):
    """
    Perform detailed head-to-head analysis between two specific drivers.
    Shows win/loss record, battle timeline, and detailed battle information.
    
    Args:
        history_df: DataFrame containing teammate battle history
        driver1: Name of first driver to compare
        driver2: Name of second driver to compare
    """
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for analysis")
        return
    
    try:
        # Find all direct battles between the two drivers
        # A battle exists when one driver beats the other as teammates
        direct_battles = history_df[
            ((history_df['winnerName'] == driver1) & (history_df['loserName'] == driver2)) |
            ((history_df['winnerName'] == driver2) & (history_df['loserName'] == driver1))
        ].sort_values('date')  # Sort chronologically
        
        # Handle case where drivers were never teammates
        if len(direct_battles) == 0:
            st.warning("❌ These drivers were never teammates")
            
            # Show helpful suggestion section for finding battles
            with st.expander("💡 Find battles for these drivers"):
                # Get all constructors each driver raced for
                d1_battles = history_df[
                    (history_df['winnerName'] == driver1) | (history_df['loserName'] == driver1)
                ]['constructor'].unique()
                d2_battles = history_df[
                    (history_df['winnerName'] == driver2) | (history_df['loserName'] == driver2)
                ]['constructor'].unique()
                
                # Display career information
                st.write(f"**{driver1}** raced for: {', '.join(d1_battles)}")
                st.write(f"**{driver2}** raced for: {', '.join(d2_battles)}")
                
                # Check if they shared any teams (but not as teammates)
                common_teams = set(d1_battles) & set(d2_battles)
                if common_teams:
                    st.info(f"💡 They both raced for: {', '.join(common_teams)} (but not as teammates)")
            return
        
        # Calculate head-to-head statistics
        driver1_wins = len(direct_battles[direct_battles['winnerName'] == driver1])
        driver2_wins = len(direct_battles[direct_battles['winnerName'] == driver2])
        total_battles = len(direct_battles)
        
        # Display key metrics in four-column layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Driver 1's win count
        with col1:
            st.metric(f"{driver1} Wins", driver1_wins)
        # Column 2: Driver 2's win count
        with col2:
            st.metric(f"{driver2} Wins", driver2_wins)
        # Column 3: Total battles between them
        with col3:
            st.metric("Total Battles", total_battles)
        # Column 4: Driver 1's win percentage
        with col4:
            win_rate = driver1_wins / total_battles if total_battles > 0 else 0
            st.metric(f"{driver1} Win Rate", f"{win_rate*100:.1f}%")
        
        # Create battle timeline visualization if battles exist
        if len(direct_battles) > 0:
            # Prepare timeline data with cumulative win tracking
            timeline_data = []
            cumulative_d1 = 0  # Running count of driver1 wins
            cumulative_d2 = 0  # Running count of driver2 wins
            
            # Process each battle chronologically
            for _, battle in direct_battles.iterrows():
                # Update cumulative win counts
                if battle['winnerName'] == driver1:
                    cumulative_d1 += 1
                else:
                    cumulative_d2 += 1
                
                # Safe numeric conversion of battle metrics
                margin = pd.to_numeric(battle['margin'], errors='coerce')  # Victory margin
                delta = pd.to_numeric(battle['delta'], errors='coerce')   # Rating change
                
                # Add data point to timeline
                timeline_data.append({
                    'Date': pd.to_datetime(battle['date']),
                    'Race': battle.get('raceName', 'Unknown'),
                    f'{driver1} Wins': cumulative_d1,
                    f'{driver2} Wins': cumulative_d2,
                    'Winner': battle['winnerName'],
                    'Margin': margin if pd.notna(margin) else 0,
                    'Rating Change': delta if pd.notna(delta) else 0
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create interactive timeline chart showing cumulative wins over time
            fig = go.Figure()
            # Add line for driver1's cumulative wins
            fig.add_trace(go.Scatter(
                x=timeline_df['Date'],
                y=timeline_df[f'{driver1} Wins'],
                name=driver1,
                mode='lines+markers'  # Show both line and data points
            ))
            # Add line for driver2's cumulative wins
            fig.add_trace(go.Scatter(
                x=timeline_df['Date'],
                y=timeline_df[f'{driver2} Wins'],
                name=driver2,
                mode='lines+markers'
            ))
            
            # Configure chart layout
            fig.update_layout(
                title=f'{driver1} vs {driver2} - Battle Timeline',
                xaxis_title='Date',
                yaxis_title='Cumulative Wins',
                height=400
            )
            
            # Display the interactive timeline chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Battle details table with safe numeric handling
            st.subheader("Battle Details")
            battle_details = direct_battles.copy()
            
            # Safe column preparation - only include columns that exist
            display_columns = ['date', 'raceName', 'winnerName']  # Essential columns
            optional_columns = ['margin', 'startDiff', 'delta']  # Optional metrics
            
            # Add optional columns if they exist in the data
            for col in optional_columns:
                if col in battle_details.columns:
                    display_columns.append(col)
            
            battle_display = battle_details[display_columns].copy()
            
            # Safe numeric formatting with error handling
            if 'margin' in battle_display.columns:
                # Format victory margin to 2 decimal places
                battle_display['Victory Margin'] = pd.to_numeric(battle_display['margin'], errors='coerce').round(2)
            if 'startDiff' in battle_display.columns:
                # Format starting grid position difference
                battle_display['Grid Deficit'] = pd.to_numeric(battle_display['startDiff'], errors='coerce')
            if 'delta' in battle_display.columns:
                # Format Elo rating change to 1 decimal place
                battle_display['Rating Change'] = pd.to_numeric(battle_display['delta'], errors='coerce').round(1)
            
            # Select final columns for display table
            final_columns = ['date', 'raceName', 'winnerName']
            if 'Victory Margin' in battle_display.columns:
                final_columns.append('Victory Margin')
            if 'Grid Deficit' in battle_display.columns:
                final_columns.append('Grid Deficit')
            if 'Rating Change' in battle_display.columns:
                final_columns.append('Rating Change')
            
            # Rename columns for better presentation
            battle_display.rename(columns={'winnerName': 'Winner'}, inplace=True)
            if 'winnerName' in final_columns:
                final_columns[final_columns.index('winnerName')] = 'Winner'

            # Format dates and sort by most recent first
            battle_display['date'] = pd.to_datetime(battle_display['date']).dt.date
            battle_display.sort_values("date", ascending=False, inplace=True)
            
            # Display the battle details table
            safe_display(
                battle_display[final_columns],
                use_container_width=True
            )
    
    except Exception as e:
        # Handle any errors in head-to-head analysis
        st.error(f"Error in head-to-head analysis: {e}")


def teammate_battle_rankings(history_df, min_battles=5):
    """
    Show all-time teammate battle records with comprehensive statistics.
    Displays the most dominant teammate pairings based on win rates.
    
    Args:
        history_df: DataFrame containing teammate battle history
        min_battles: Minimum number of battles required for inclusion (default: 5)
    """
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for teammate battle rankings")
        return
    
    try:
        # Dictionary to store battle records for each driver pairing
        battle_records = defaultdict(lambda: defaultdict(int))
        # Dictionary to store additional details about each pairing
        battle_details = defaultdict(lambda: defaultdict(list))
        
        # Process each battle in the dataset
        for _, row in history_df.iterrows():
            winner = row.get('winnerName')
            loser = row.get('loserName')
            
            # Skip battles with missing driver names
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Create consistent pairing key (alphabetical order for consistency)
            pair = tuple(sorted([winner, loser]))
            
            # Update battle statistics
            battle_records[pair]['total'] += 1      # Total battles between this pair
            battle_records[pair][winner] += 1       # Wins for the winner
            
            # Collect additional battle context safely
            constructor = row.get('constructor', 'Unknown')
            year = row.get('year')
            # Store constructor information (team they raced for)
            if pd.notna(constructor):
                battle_details[pair]['constructor'].append(constructor)
            # Store year information
            if pd.notna(year):
                battle_details[pair]['years'].append(int(year))
        
        # Convert battle records to display format
        ranking_data = []
        for (driver1, driver2), record in battle_records.items():
            total = record['total']
            d1_wins = record.get(driver1, 0)
            d2_wins = record.get(driver2, 0)
            
            # Only include pairings with sufficient battles for statistical significance
            if total >= min_battles:
                # Determine which driver is dominant in the pairing
                if d1_wins > d2_wins:
                    dominant, dominated = driver1, driver2
                    dom_wins, dom_rate = d1_wins, d1_wins/total
                else:
                    dominant, dominated = driver2, driver1
                    dom_wins, dom_rate = d2_wins, d2_wins/total
                
                # Safe extraction of additional details
                constructors = list(set(battle_details[(driver1, driver2)]['constructor']))
                years_list = battle_details[(driver1, driver2)]['years']
                # Create year range string
                years = f"{min(years_list)}-{max(years_list)}" if years_list else "Unknown"
                
                # Add pairing data to ranking list
                ranking_data.append({
                    'Pairing': f"{driver1} vs {driver2}",
                    'Dominant Driver': dominant,
                    'Record': f"{dom_wins}-{total-dom_wins}",  # W-L format
                    'Win Rate': f"{dom_rate*100:.1f}%",
                    'Total Battles': total,
                    'Teams': ', '.join(constructors[:2]) if constructors else 'Unknown',  # Limit to 2 teams for display
                    'Years': years
                })
        
        # Handle case where no pairings meet minimum battle requirement
        if not ranking_data:
            st.info(f"📊 No teammate pairings found with minimum {min_battles} battles. Try reducing the minimum battles requirement.")
            return
        
        # Sort by dominance (highest win rate first)
        ranking_df = pd.DataFrame(ranking_data)
        # Create numeric win rate column for sorting
        ranking_df['Win Rate Numeric'] = ranking_df['Win Rate'].str.rstrip('%').astype(float)
        
        # Display most dominant teammate pairings
        st.write("**Most Dominant Teammate Battles:**")
        dominant_df = ranking_df.sort_values('Win Rate Numeric', ascending=False).head(10)
        
        # Display rankings table
        safe_display(
            dominant_df[['Pairing', 'Dominant Driver', 'Record', 'Win Rate', 'Teams', 'Years']],
            use_container_width=True
        )
    
    except Exception as e:
        # Handle any errors in teammate battle rankings
        st.error(f"Error in teammate battle rankings: {e}")


def competitive_pairings(history_df, min_battles=8):
    """
    Show most competitive (close to 50-50) teammate pairings.
    Identifies pairings where neither driver has a significant advantage.
    
    Args:
        history_df: DataFrame containing teammate battle history
        min_battles: Minimum number of battles required for inclusion (default: 8)
    """
    # Handle empty data case
    if history_df.empty:
        st.warning("No data available for competitive pairings analysis")
        return
    
    try:
        # Dictionary to store battle records for each pairing
        battle_records = defaultdict(lambda: defaultdict(int))
        
        # Process each battle to build pairing statistics
        for _, row in history_df.iterrows():
            winner = row.get('winnerName')
            loser = row.get('loserName')
            
            # Skip battles with missing driver information
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Create consistent pairing key (alphabetical order)
            pair = tuple(sorted([winner, loser]))
            # Update battle statistics
            battle_records[pair]['total'] += 1
            battle_records[pair][winner] += 1
        
        # Analyze competitiveness of each pairing
        competitive_data = []
        for (driver1, driver2), record in battle_records.items():
            total = record['total']
            d1_wins = record.get(driver1, 0)
            d2_wins = record.get(driver2, 0)
            
            # Only consider pairings with sufficient battles for meaningful competition analysis
            if total >= min_battles:
                # Calculate win rate of the leading driver
                win_rate = max(d1_wins, d2_wins) / total
                # Calculate competitiveness score: closer to 0.5 = more competitive
                competitiveness = abs(win_rate - 0.5)  # Distance from perfect 50-50 split
                
                competitive_data.append({
                    'Pairing': f"{driver1} vs {driver2}",
                    'Record': f"{d1_wins}-{d2_wins}",
                    'Leader Win Rate': f"{win_rate*100:.1f}%",
                    'Competitiveness': competitiveness,  # Lower = more competitive
                    'Total Battles': total
                })
        
        # Handle case where no competitive pairings found
        if not competitive_data:
            st.info(f"📊 No competitive pairings found with minimum {min_battles} battles. Try reducing the minimum battles requirement.")
            return
        
        # Sort by competitiveness (lower competitiveness score = more evenly matched)
        comp_df = pd.DataFrame(competitive_data).sort_values('Competitiveness').head(10)
        
        # Display competitive pairings table
        safe_display(
            comp_df[['Pairing', 'Record', 'Leader Win Rate', 'Total Battles']],
            use_container_width=True
        )
        
        # Create visualization with comprehensive error handling
        try:
            if len(comp_df) > 0:
                # Prepare data for plotting
                comp_df_plot = comp_df.copy()
                # Convert win rate string to numeric for plotting
                comp_df_plot['Win Rate Numeric'] = comp_df_plot['Leader Win Rate'].str.rstrip('%').astype(float)
                
                # Create bar chart showing competitiveness
                fig = px.bar(
                    comp_df_plot,
                    x='Pairing',
                    y='Win Rate Numeric',
                    title='Most Competitive Teammate Battles (Closest to 50-50)',
                    color='Total Battles',  # Color bars by number of battles
                    labels={'Win Rate Numeric': 'Leader Win Rate (%)'}
                )
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45)
                # Add reference line at 50% (perfect competition)
                fig.add_hline(y=50, line_dash="dash", annotation_text="Perfect Competition")
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
        except Exception as viz_error:
            # Handle visualization errors gracefully
            st.warning(f"Could not create visualization: {viz_error}")
    
    except Exception as e:
        # Handle any errors in competitive pairings analysis
        st.error(f"Error in competitive pairings analysis: {e}")
