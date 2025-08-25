# Import necessary libraries for the F1 statistics dashboard
import time  # Used for sleep delays in UI interactions

import streamlit as st  # Main web app framework
import pandas as pd  # Data manipulation and analysis library

# Import custom modules for Azure cloud storage functionality
from azure_functions import *

# Import individual page modules for different analytics sections
from streamlit_pages.driver_ranking import driver_rankings_page
from streamlit_pages.performance_analytics import performance_analytics_page
from streamlit_pages.advanced_stats import advanced_stats_page
from streamlit_pages.head_to_head import head_to_head_page
from streamlit_pages.race_statistics import race_statistics_page
from streamlit_pages.track_analytics import track_analytics_page
from streamlit_pages.reliability_analysis import reliability_analysis_page
from streamlit_pages.about import display_about_page

# Import utility functions for data processing
from util import *
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('secret.env')

# Commented out import for potential future ELO ranking functionality
# from pages.elo_list import elo_list

# Initialize Azure Blob Storage container client for cloud data access
CONTAINER_CLIENT = get_blob_object()


@st.cache_data
def read_data() -> pd.DataFrame:
    """
    Read all user stats from azure and filters out all bots
    
    This function loads the core datasets needed for the F1 analytics dashboard:
    - ELO ratings data for driver skill rankings
    - Historical race data for performance analysis
    - Results data for race outcomes and statistics
    
    Returns:
        tuple: Three DataFrames containing (elo_data, history_data, results_data)
    """
    # Commented out Azure blob storage data loading (for production deployment)
    with read_file_from_blob("RESULT_DATA/elo.csv", CONTAINER_CLIENT) as file:
        df = pd.read_csv(file)
    
    with read_file_from_blob("RESULT_DATA/history.csv", CONTAINER_CLIENT) as file:
        history_df = pd.read_csv(file)
    
    with read_file_from_blob("RESULT_DATA/results.csv", CONTAINER_CLIENT) as file:
        results_df = pd.read_csv(file)

    # Load ELO ratings data from local CSV file
    # Contains driver skill ratings based on historical performance
    # with open("data/elo.csv", "r", encoding="utf-8") as file:
    #     df = pd.read_csv(file)
    
    # # Load historical race data from local CSV file
    # # Contains detailed race-by-race performance metrics and calculations
    # with open("data/history.csv", "r", encoding="utf-8") as file:
    #     history_df = pd.read_csv(file)

    # # Load race results data from local CSV file
    # # Contains basic race outcome information (positions, DNFs, etc.)
    # with open("data/results.csv", "r", encoding="utf-8") as file:
    #     results_df = pd.read_csv(file)

    return df, history_df, results_df


# Configure Streamlit page settings for the F1 dashboard
st.set_page_config(
    page_title="FormHub",  # Browser tab title
    page_icon="🏎️",        # Browser tab icon (racing car emoji)
    layout="wide"          # Use wide layout for better data visualization
)


def preprocess_dataframes(history_df, rating_df, results_df=None):
    """
    Comprehensive preprocessing to fix all data type issues
    
    This function ensures all DataFrames have correct data types for:
    - Date columns: Converted to datetime objects for time-series analysis
    - Numeric columns: Converted to appropriate numeric types for calculations
    - Handles missing values and data type inconsistencies
    
    Args:
        history_df: Historical race performance data
        rating_df: Driver ELO ratings data
        results_df: Basic race results data (optional)
    
    Returns:
        tuple: Preprocessed DataFrames with corrected data types
    """

    # Create copies to avoid modifying original DataFrames
    history_df = history_df.copy()
    
    # Convert date columns to datetime objects for proper time-series operations
    date_columns = ['date']
    for col in date_columns:
        if col in history_df.columns:
            history_df[col] = pd.to_datetime(history_df[col], errors='coerce')
    
    # Define all numeric columns in history_df that need type conversion
    # Includes race positions, performance metrics, ELO calculations, and race statistics
    numeric_columns = [
        'year', 'round', 'finalPosition', 'startPosition',  # Basic race info
        'positionChange', 'endDiffTeammate', 'startDiffTeammate',  # Position metrics
        'lapsCompleted', 'dnf', 'delta', 'K_eff', 'E_w', 'S_w',  # Performance metrics
        'Rw_prev', 'Rl_prev', 'Rw_new', 'Rl_new', 'margin',  # ELO rating calculations
        'comeback_bonus', 'track_adjustment', 'lap_completion_diff',  # Bonus adjustments
        'dnf_scale_factor', 'positionChange_W', 'positionChange_L',  # Scaling factors
        'lapsCompleted_W', 'lapsCompleted_L', 'race_distance'  # Completion metrics
    ]
    
    # Apply numeric conversion using utility function
    history_df = preprocess_numeric_columns(history_df, numeric_columns)
    
    # Process rating DataFrame - contains driver ELO ratings and match statistics
    rating_df = rating_df.copy()
    rating_numeric = ['rating', 'matches', 'confidence']  # Core rating metrics
    rating_df = preprocess_numeric_columns(rating_df, rating_numeric)
    
    # Process results DataFrame if provided (optional parameter)
    if results_df is not None:
        results_df = results_df.copy()
        # Convert date column in results data
        if 'date' in results_df.columns:
            results_df['date'] = pd.to_datetime(results_df['date'], errors='coerce')
        
        # Define numeric columns specific to results data
        results_numeric = ['year', 'round', 'finalPosition', 'startPosition', 
                          'positionChange', 'lapsCompleted', 'dnf']
        results_df = preprocess_numeric_columns(results_df, results_numeric)
    
    return history_df, rating_df, results_df


@st.dialog("Admin Login")
def admin_login():
    """
    Pop up box for admin login
    
    Creates a modal dialog for admin authentication to access advanced features.
    Uses Streamlit's session state to maintain login status across page interactions.
    Simple password-based authentication for demonstration purposes.
    """

    st.write(f"Please enter password below")
    # Create password input field with hidden text
    pwd = st.text_input("Password", type="password", )
    
    if st.button("Submit"):
        # Simple password check (in production, use proper authentication)
        if pwd == os.getenv("ADMIN_PASSWORD"):
            # Set admin flag in session state for persistent login
            st.session_state["admin"] = True
            st.success("Success! You can now see advanced statistics page.")
            time.sleep(2)  # Brief pause to show success message
            st.rerun()     # Refresh page to update UI with admin privileges
        else:
            st.error("Incorrect password")


@st.cache_data()
def get_latest_race(res):
    """
    Get the most recent race information from results data
    
    Sorts the results DataFrame by date and returns formatted string
    with the year and race name of the most recent race.
    Used to display data currency information to users.
    
    Args:
        res: Results DataFrame containing race data
    
    Returns:
        str: Formatted string with latest race info (e.g., "2024 Abu Dhabi Grand Prix")
    """
    res.sort_values("date", inplace=True, ascending=False)  # Sort by most recent first
    return f"{res.iloc[0]['year']} {res.iloc[0]['raceName']}"  # Return latest race info


def main():
    """
    Main application function that orchestrates the entire F1 analytics dashboard
    
    Handles:
    - Data loading and preprocessing
    - User interface setup and navigation
    - Session state management for admin access and about page
    - Page routing to different analytics modules
    - Initial app state and welcome screen
    """

    # Load all required datasets using cached function for performance
    rating_df, history_df, results_df = read_data()
    
    # Preprocess all DataFrames to ensure correct data types
    history_df, rating_df, results_df = preprocess_dataframes(history_df, rating_df, results_df)

    # Set up main page header and data currency information
    st.title("🏎️ FormHub - Formula 1 Analytics Hub")
    # Display badge showing the latest race data available
    st.badge(f"Data available till {get_latest_race(results_df)}", color="blue", icon=":material/update:")
    st.sidebar.title("Navigation")

    # Initialize session state variables for application state management
    if "admin" not in st.session_state:
        st.session_state["admin"] = False  # Default to non-admin user
    
    if "about" not in st.session_state:
        st.session_state["about"] = True   # Show about page on first visit
    
    # Display welcome/about page for first-time users
    if st.session_state["about"]:
        display_about_page()  # Show information about the dashboard
        # Button to proceed to main application
        if st.button("Let the fun begin", type="primary"):
            st.session_state["about"] = False
            st.rerun()
        st.stop()  # Don't render rest of the app while showing about page
    
    # Define available pages for regular users
    page_list = ["Driver Rankings", "Track Analytics", "Head-to-Head", "Performance Analytics", 
         "Reliability Analysis", "Race Statistics"]

    # Add admin-only pages if user has admin privileges
    if st.session_state["admin"]:
        page_list += ["Advanced Stats"]
    else:
        # Show admin login button for non-admin users
        if st.sidebar.button("Admin", type="secondary"):
            admin_login()

    # Create page selection dropdown in sidebar
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        page_list, 
    )

    # Help button to return to about page for user guidance
    if st.sidebar.button("Help"):
        st.session_state["about"] = True
        st.rerun()

    # Route to appropriate page based on user selection
    # Each page function receives the preprocessed DataFrames as parameters
    if page == "Driver Rankings":
        driver_rankings_page(history_df, rating_df)
    elif page == "Track Analytics":
        track_analytics_page(history_df, rating_df, results_df)
    elif page == "Head-to-Head":
        head_to_head_page(history_df, rating_df)
    elif page == "Performance Analytics":
        performance_analytics_page(history_df, rating_df, results_df)
    elif page == "Reliability Analysis":
        reliability_analysis_page(history_df, rating_df)
    elif page == "Race Statistics":
        race_statistics_page(history_df, rating_df, results_df)
    elif page == "Advanced Stats":
        advanced_stats_page(history_df, rating_df)


# Application entry point - runs main function when script is executed directly
if __name__ == "__main__":
    main()
