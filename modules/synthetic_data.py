import streamlit as st

def show():
    st.title("🎲 Synthetic Data Generation")
    
    st.markdown("""
    ### Data is the New Gold
    
    Running out of training data? Or real data is too private (PII)?
    **Synthetic Data** is created by AI, for AI.
    """)
    
    tabs = st.tabs([
        "🏭 Why Synthetic?",
        "🧪 Gen Techniques",
        "👨‍💻 Code Demo"
    ])
    
    # TAB 1: Why
    with tabs[0]:
        st.header("🏭 Why Synthetic Data?")
        
        st.markdown("""
        1. **Privacy:** Generate "fake patients" that statistically look like real ones but preserve privacy (HIPAA).
        2. **Scarcity:** Generate rare edge cases (e.g., specific fraud patterns) that rarely happen in nature.
        3. **Cost:** Cheaper to generate data than to pay humans to label it.
        4. **Bias Correction:** Deliberately generate balanced datasets (e.g., equal diversity).
        """)

    # TAB 2: Techniques
    with tabs[1]:
        st.header("🧪 Generation Techniques")
        
        st.subheader("1. LLM-Based Generation")
        st.markdown("Ask GPT-4 to generate examples.")
        st.markdown("> *Promp: Generate 10 examples of customer complaints about a broken toaster, in JSON format.*")
        
        st.subheader("2. Self-Instruct (Alpaca method)")
        st.markdown("1. Seed: Give LLM 3 human-written examples.\n2. Ask it to generate 100 similar ones.\n3. Filter for quality.\n4. Repeat.")
        
        st.subheader("3. Evol-Instruct")
        st.markdown("Ask LLM to rewrite simple instructions to be more complex/difficult.")
        
        st.subheader("4. Faker Libraries")
        st.markdown("Traditional code for generating names, emails, dates.")

    # TAB 3: Code
    with tabs[2]:
        st.header("👨‍💻 Generating Test Cases")
        
        st.code('''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# Define Schema
class CustomerProfile(BaseModel):
    name: str
    age: int
    complaint: str
    sentiment: str = Field(description="angry, sad, or neutral")

class SyntheticDataset(BaseModel):
    profiles: List[CustomerProfile]

# Create Generator
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(SyntheticDataset)

# Generate
prompt = "Generate 5 synthetic customer profiles for a telecom company."
dataset = structured_llm.invoke(prompt)

# Use Data
for p in dataset.profiles:
    print(f"{p.name} ({p.age}): {p.complaint}")
        ''', language="python")
        
        st.success("You just created a perfectly structured dataset for testing your support bot!")
