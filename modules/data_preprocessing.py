import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processing import clean_messy_data

def show():
    st.title("🧹 Module 2: Getting Data Ready")
    
    st.markdown("""
    ### Before AI can learn, we need to clean and prepare the data!
    
    *Think of it like preparing ingredients before cooking* 🍳
    """)
    
    tabs = st.tabs(["🥗 Why Clean Data?", "❓ Missing Values", "📏 Scaling Numbers", "🏷️ Categories", "🎮 Practice"])
    
    # TAB 1: Why Clean Data
    with tabs[0]:
        st.header("🥗 Why Does Data Need Cleaning?")
        
        st.markdown("""
        ### The Cooking Analogy 🍳
        
        Imagine you're teaching someone to cook by showing them recipes:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            ### 🗑️ Bad Ingredients (Dirty Data)
            
            - 🥚 Eggs: "two" (text instead of number)
            - 🧈 Butter: ??? (missing amount)
            - 🌡️ Oven: 356°F or 180°C? (inconsistent units)
            - 📝 Instructions: "bak for 20 minuts" (typos)
            
            **Result:** Confused cook, burnt cake! 😢
            """)
            
        with col2:
            st.success("""
            ### ✨ Clean Ingredients (Good Data)
            
            - 🥚 Eggs: 2
            - 🧈 Butter: 100g
            - 🌡️ Oven: 180°C
            - 📝 Instructions: "Bake for 20 minutes"
            
            **Result:** Perfect cake! 🎂
            """)
        
        st.info("""
        ### 💡 Key Insight
        
        **"Garbage In = Garbage Out"**
        
        Even the smartest AI can't learn from messy data!
        
        📊 Data scientists spend **80% of their time** just cleaning data.
        """)
        
        st.subheader("Common Data Problems")
        
        problems = [
            ("❓ Missing Values", "Empty cells in your data", "A form where someone skipped their phone number"),
            ("🔢 Wrong Format", "Numbers as text or wrong types", "Age written as 'twenty-five' instead of 25"),
            ("📏 Different Scales", "Numbers in different ranges", "Salary: $50,000 vs Age: 25 (huge difference!)"),
            ("🏷️ Inconsistent Categories", "Same thing, different names", "'USA', 'United States', 'U.S.A.' = same country"),
            ("📊 Outliers", "Extreme values that don't fit", "Age: 25, 30, 28, 500 (probably a typo!)"),
        ]
        
        for emoji_name, what, example in problems:
            with st.expander(emoji_name):
                st.markdown(f"**What it means:** {what}")
                st.markdown(f"**Example:** {example}")
    
    # TAB 2: Missing Values
    with tabs[1]:
        st.header("❓ Handling Missing Values")
        
        st.markdown("""
        ### The Survey Analogy 📋
        
        Imagine you collected surveys and some people left questions blank:
        """)
        
        # Sample data with missing values
        survey_data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
            'Age': [25, None, 35, 28, None],
            'City': ['NYC', 'LA', None, 'Chicago', 'NYC'],
            'Rating': [5, 4, 5, None, 3]
        })
        
        st.dataframe(survey_data.style.applymap(lambda x: 'background-color: #ffcccc' if pd.isna(x) else ''))
        st.caption("Red cells = Missing values (None)")
        
        st.markdown("---")
        st.subheader("What Can We Do?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🗑️ Option 1: Remove
            
            **Delete rows with missing data**
            
            ✅ Simple and clean
            ❌ Lose good data too
            
            *Like throwing away a survey because ONE question is blank*
            """)
            
        with col2:
            st.markdown("""
            ### 📊 Option 2: Fill with Average
            
            **Replace blanks with the average**
            
            ✅ Keeps all data
            ❌ Might not be accurate
            
            *Like guessing someone's age is 30 (the average)*
            """)
            
        with col3:
            st.markdown("""
            ### 🔮 Option 3: Smart Guess
            
            **Use patterns to predict**
            
            ✅ Most accurate
            ❌ More complex
            
            *Like guessing age based on their job and hobbies*
            """)
        
        st.markdown("---")
        st.subheader("🎮 Interactive Demo")
        
        strategy = st.selectbox(
            "How should we handle missing Ages?",
            ["Do nothing", "Remove rows with missing", "Fill with average (mean)", "Fill with most common (mode)"]
        )
        
        demo_ages = pd.Series([25, None, 35, 28, None])
        
        if strategy == "Remove rows with missing":
            result = demo_ages.dropna()
            st.success(f"**Result:** Kept only complete rows: {list(result.values)}")
            st.warning("⚠️ Lost 2 out of 5 rows (40% of data!)")
            
        elif strategy == "Fill with average (mean)":
            avg = demo_ages.mean()
            result = demo_ages.fillna(avg)
            st.success(f"**Result:** Filled blanks with average ({avg:.0f}): {list(result.values)}")
            
        elif strategy == "Fill with most common (mode)":
            st.info("All ages appear once, so we'd pick any (e.g., 25)")
            result = demo_ages.fillna(25)
            st.success(f"**Result:** Filled blanks with 25: {list(result.values)}")
        else:
            st.dataframe(pd.DataFrame({'Ages': demo_ages}))
    
    # TAB 3: Scaling
    with tabs[2]:
        st.header("📏 Scaling Numbers")
        
        st.markdown("""
        ### The Money Comparison Problem 💰
        
        Imagine comparing:
        - 🏠 House Price: **$500,000**
        - 🛏️ Number of Bedrooms: **3**
        
        The house price is 166,667x bigger! 
        
        **AI might think price is 166,667x more important.**
        
        *That's not fair!*
        """)
        
        st.subheader("The Solution: Put Everything on the Same Scale")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📐 0 to 1 Scaling (Min-Max)
            
            **Like converting grades to percentages**
            
            Before: Scores 50, 60, 70, 80, 90
            
            After: 0.0, 0.25, 0.5, 0.75, 1.0
            
            *Smallest = 0, Largest = 1*
            """)
            
        with col2:
            st.markdown("""
            ### 📊 Z-Score Scaling
            
            **Like grading on a curve**
            
            "How far from average are you?"
            
            - Average student = 0
            - Better than average = positive
            - Worse than average = negative
            """)
        
        st.markdown("---")
        st.subheader("🎮 See Scaling in Action")
        
        # Demo data
        prices = [100000, 200000, 300000, 400000, 500000]
        bedrooms = [1, 2, 3, 4, 5]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Before Scaling")
            fig1 = px.bar(x=['Price', 'Bedrooms'], y=[np.mean(prices), np.mean(bedrooms)], 
                         title="Average Values (Not Comparable!)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("### After 0-1 Scaling")
            # Scale to 0-1
            prices_scaled = [(p - min(prices)) / (max(prices) - min(prices)) for p in prices]
            beds_scaled = [(b - min(bedrooms)) / (max(bedrooms) - min(bedrooms)) for b in bedrooms]
            
            fig2 = px.bar(x=['Price', 'Bedrooms'], y=[np.mean(prices_scaled), np.mean(beds_scaled)],
                         title="Average Values (Now Comparable!)")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.success("""
        ### 💡 After scaling:
        
        - Both features are between 0 and 1
        - AI treats them equally important (unless patterns say otherwise)
        - Fair comparison!
        """)
    
    # TAB 4: Categories
    with tabs[3]:
        st.header("🏷️ Converting Categories to Numbers")
        
        st.markdown("""
        ### The Problem
        
        AI only understands **numbers**. It can't read words like "Red", "Blue", "Green".
        
        We need to convert categories to numbers!
        """)
        
        st.subheader("Method 1: Label Encoding")
        
        st.info("""
        **Simple numbering**
        
        | Color | Number |
        |-------|--------|
        | Red | 0 |
        | Blue | 1 |
        | Green | 2 |
        
        ⚠️ **Problem:** AI might think Green (2) is "bigger" than Red (0)
        
        *Good for categories with natural order (Small, Medium, Large)*
        """)
        
        st.subheader("Method 2: One-Hot Encoding")
        
        st.success("""
        **Create separate columns for each category**
        
        | Color | Is_Red | Is_Blue | Is_Green |
        |-------|--------|---------|----------|
        | Red | 1 | 0 | 0 |
        | Blue | 0 | 1 | 0 |
        | Green | 0 | 0 | 1 |
        
        ✅ **No fake ordering problem!**
        
        *Like asking 3 yes/no questions instead of 1*
        """)
        
        st.markdown("---")
        st.subheader("🎮 Real Example")
        
        st.markdown("**Customer data with categories:**")
        
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Carol'],
            'Gender': ['Female', 'Male', 'Female'],
            'City': ['NYC', 'LA', 'NYC']
        })
        st.dataframe(df)
        
        st.markdown("**After One-Hot Encoding:**")
        
        df_encoded = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Carol'],
            'Gender_Female': [1, 0, 1],
            'Gender_Male': [0, 1, 0],
            'City_NYC': [1, 0, 1],
            'City_LA': [0, 1, 0]
        })
        st.dataframe(df_encoded)
        
        st.success("Now AI can understand! Each column is just 0 or 1.")
    
    # TAB 5: Practice
    with tabs[4]:
        st.header("🎮 Practice: Clean This Data!")
        
        st.markdown("""
        ### Your Mission: Help AI learn from this messy data! 🎯
        """)
        
        # Messy dataset
        messy_data = pd.DataFrame({
            'Student': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
            'Age': [22, None, 25, 'twenty', 23],
            'Score': [85, 90, None, 78, 95],
            'Grade': ['A', 'A', 'B', 'C', 'A'],
            'City': ['New York', 'LA', 'NYC', 'Los Angeles', 'new york']
        })
        
        st.markdown("### 😱 The Messy Data:")
        st.dataframe(messy_data)
        
        st.markdown("### 🔍 Find the Problems!")
        
        problems = st.multiselect(
            "What problems do you see? (Select all that apply)",
            [
                "Missing values (None)",
                "Age 'twenty' is text instead of number",
                "NYC and New York are the same city",
                "LA and Los Angeles are the same",
                "Inconsistent capitalization (new york)"
            ]
        )
        
        if len(problems) >= 3:
            st.success(f"🎉 Great job! You found {len(problems)} problems!")
            
            st.markdown("### ✅ How to Fix Each Problem:")
            
            fixes = {
                "Missing values (None)": "Fill with average or remove rows",
                "Age 'twenty' is text instead of number": "Convert 'twenty' to 20",
                "NYC and New York are the same city": "Standardize to one name (e.g., 'New York')",
                "LA and Los Angeles are the same": "Standardize to one name (e.g., 'Los Angeles')",
                "Inconsistent capitalization (new york)": "Convert all to same case (e.g., Title Case)"
            }
            
            for problem in problems:
                st.info(f"**{problem}** → {fixes.get(problem, 'Fix it!')}")
            
            st.markdown("### 🧹 Clean Data:")
            
            # Use our new tested utility function!
            clean_data = clean_messy_data(messy_data)
            
            st.dataframe(clean_data)
            st.balloons()
