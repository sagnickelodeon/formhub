import streamlit as st

def display_about_page():
    text = """
    # Welcome to **FormHub**  
    **Formula 1 Teammate Analytics Hub** - an all-in-one web app that lets you explore how F1 drivers, teams, and races stack up when the playing field is level.  
    Every metric you see is calculated only from head-to-head battles between teammates sharing the same car - the fairest way to separate driver skill from machinery. This contains all races data from 1982 to the present, which aligns with a key turning point in F1 history: in 1982, teams were first required to commit to the entire championship season, ending the era of selective race entries. That reform helped standardize competition and laid the groundwork for comparing teammates across a full season - just like this site does.

    ---

    ## Why Teammates?

    - Two drivers in identical cars remove the “which car was faster?” debate.  
    - Each race weekend provides a matched-pair experiment, ideal for statistical models.  
    - Linking thousands of these battles across seasons creates a giant comparison web that connects Fangio to Verstappen on the same scale.

    ---

    ## 🧭 Feature Tour

    ### 1. 🏆 Driver Rankings & Ratings

    - Live Elo ladder that updates after every intra-team battle.  
    - Confidence indicator grows with race count so rookies aren't overrated.  
    - Rating distribution histograms to spot outliers and midfield clusters.  
    - Timeline charts to watch a driver's rating rise or fall across seasons.

    ---

    ### 2. ⚔️ Head-to-Head Arena

    Pick any two drivers and instantly see:  
    - Win/loss record as teammates  
    - Cumulative score progression by race  
    - Victory margins, grid-start deficits, and Elo swings for each duel  
    - League tables of the most dominant and most evenly-matched teammate pairings in F1 history

    ---

    ### 3. 📈 Performance Analytics

    - **Comeback Kings** - who gains the most places after starting deep in the field  
    - **Race Craft vs Qualifying** - scatter plot shows late-race magicians and pure one-lap specialists  
    - **Form Guide** - rolling win-rate and rating-trend bar chart for the last X races  
    - **Consistency Meter** - volatility of race-by-race Elo changes highlights rock-solid performers vs wildcards

    ---

    ### 4. 🏁 Race Statistics Suite

    - **Grand Prix Records** - for any circuit, discover its most successful drivers, drama factor, and active years  
    - **Season Dashboards** - cumulative win charts that reveal championship swings round-by-round  
    - **Race Characteristics** - margins vs rating gains, drama trends, DNF rates, and comeback frequencies over decades

    ---

    ### 5. 🔧 Reliability Analyzer

    - **Driver DNFs** - early vs late retirements, reliability scores, and outlier scatter  
    - **Lap Completion** - average completion %, consistency score, and total laps banked  
    - **Team Reliability** - constructor DNF rates with year-by-year trend lines

    """
    st.markdown(text)

    return