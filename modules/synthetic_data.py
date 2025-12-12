import streamlit as st
import pandas as pd
import json
import random

def show():
    st.title("🎲 Synthetic Data Generation for AI")
    
    st.markdown("""
    **"Data is the new oil, but synthetic data is renewable energy."**  
    Learn how to generate high-quality training data using AI, without privacy concerns or labeling costs.
    """)
    
    tabs = st.tabs([
        "🏭 Why Synthetic Data?",
        "🧪 Generation Techniques",
        "💻 LLM-Based Generation",
        "🔧 Traditional Methods",
        "✅ Quality Control",
        "⚖️ Ethics & Risks"
    ])
    
    # TAB 1: Why Synthetic Data?
    with tabs[0]:
        st.header("🏭 The Case for Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Problems with Real Data")
            st.markdown("""
            ❌ **Privacy Concerns**
            - GDPR, HIPAA compliance
            - Can't share customer data
            - PII removal is hard
            
            ❌ **Data Scarcity**
            - Rare events (fraud, failures)
            - Long-tail scenarios
            - Expensive to collect
            
            ❌ **Labeling Costs**
            - $0.10 - $5 per label
            - Slow turnaround
            - Quality inconsistency
            
            ❌ **Bias Issues**
            - Underrepresented groups
            - Historical biases
            - Imbalanced classes
            """)
        
        with col2:
            st.subheader("Benefits of Synthetic Data")
            st.markdown("""
            ✅ **Privacy-Preserving**
            - No real PII
            - Shareable datasets
            - GDPR compliant
            
            ✅ **Infinite Scale**
            - Generate millions of examples
            - Cover edge cases
            - Instant availability
            
            ✅ **Cost-Effective**
            - $0.001 per example (LLM)
            - No human labelers
            - Automated pipeline
            
            ✅ **Bias Control**
            - Balanced demographics
            - Deliberate diversity
            - Fairness by design
            """)
        
        st.markdown("---")
        st.subheader("Real-World Use Cases")
        
        use_cases = {
            "🏥 Healthcare": "Generate synthetic patient records for ML training (HIPAA compliant)",
            "🏦 Finance": "Create fraud scenarios without exposing real transactions",
            "🤖 Chatbots": "Generate diverse customer queries for testing",
            "🚗 Autonomous Vehicles": "Simulate rare driving scenarios (accidents, weather)",
            "📝 NLP": "Augment training data for low-resource languages",
            "🎮 Gaming": "Generate NPC dialogue and behaviors"
        }
        
        for use_case, description in use_cases.items():
            st.markdown(f"**{use_case}**: {description}")

    # TAB 2: Generation Techniques
    with tabs[1]:
        st.header("🧪 Synthetic Data Generation Techniques")
        
        techniques = [
            {
                "name": "1. LLM-Based Generation",
                "description": "Use GPT-4/Claude to generate examples from prompts",
                "pros": ["High quality", "Flexible", "Handles complex schemas"],
                "cons": ["Expensive at scale", "May hallucinate", "Requires validation"],
                "best_for": "Text data, customer queries, code examples"
            },
            {
                "name": "2. Self-Instruct (Alpaca Method)",
                "description": "Seed with few examples, LLM generates more",
                "pros": ["Scales well", "Maintains consistency", "Cost-effective"],
                "cons": ["Needs good seeds", "Quality drift", "Repetitive"],
                "best_for": "Instruction datasets, Q&A pairs"
            },
            {
                "name": "3. Evol-Instruct",
                "description": "Iteratively make examples more complex",
                "pros": ["Increases difficulty", "Diverse outputs", "Controlled evolution"],
                "cons": ["Multi-step process", "Higher cost", "Needs monitoring"],
                "best_for": "Challenging test cases, reasoning tasks"
            },
            {
                "name": "4. GANs (Generative Adversarial Networks)",
                "description": "Neural network generates data, discriminator validates",
                "pros": ["Realistic images/tabular data", "No LLM needed", "Privacy-preserving"],
                "cons": ["Hard to train", "Mode collapse", "Requires expertise"],
                "best_for": "Images, time-series, tabular data"
            },
            {
                "name": "5. Faker Libraries",
                "description": "Rule-based generation (names, emails, dates)",
                "pros": ["Fast", "Cheap", "Deterministic", "No API needed"],
                "cons": ["Limited creativity", "Obvious patterns", "Not semantic"],
                "best_for": "Structured data, testing, demos"
            }
        ]
        
        for tech in techniques:
            with st.expander(f"**{tech['name']}**"):
                st.markdown(f"**Description**: {tech['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pros:**")
                    for pro in tech['pros']:
                        st.markdown(f"- ✅ {pro}")
                
                with col2:
                    st.markdown("**Cons:**")
                    for con in tech['cons']:
                        st.markdown(f"- ❌ {con}")
                
                st.info(f"**Best for**: {tech['best_for']}")

    # TAB 3: LLM-Based Generation
    with tabs[2]:
        st.header("💻 LLM-Based Synthetic Data Generation")
        
        st.subheader("Method 1: Structured Output (Recommended)")
        
        st.code("""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# Define Schema
class CustomerComplaint(BaseModel):
    name: str = Field(description="Customer name")
    product: str = Field(description="Product name")
    issue: str = Field(description="Detailed complaint")
    sentiment: str = Field(description="angry, frustrated, or neutral")
    priority: int = Field(description="1-5, where 5 is urgent")

class SyntheticDataset(BaseModel):
    complaints: List[CustomerComplaint]

# Generate
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(SyntheticDataset)

prompt = \"\"\"Generate 10 diverse customer complaints for an e-commerce company.
Include variety in:
- Product types (electronics, clothing, home goods)
- Issue types (shipping, quality, returns)
- Sentiment levels
- Demographics (implied from names)
\"\"\"

dataset = structured_llm.invoke(prompt)

# Export to JSON
import json
with open('synthetic_complaints.json', 'w') as f:
    json.dump([c.dict() for c in dataset.complaints], f, indent=2)
        """, language="python")
        
        st.markdown("---")
        st.subheader("Method 2: Few-Shot Prompting")
        
        st.code("""
prompt = \"\"\"Generate customer support conversations.

Example 1:
Customer: My order #12345 never arrived!
Agent: I apologize for the inconvenience. Let me track that for you.
Customer: It's been 2 weeks!
Agent: I see it's stuck in transit. I'll issue a refund immediately.

Example 2:
Customer: The shoes I received are the wrong size.
Agent: I'm sorry about that. What size did you order vs receive?
Customer: Ordered 9, got 7.
Agent: I'll send a prepaid return label and ship the correct size today.

Now generate 5 more similar conversations with different issues.
\"\"\"

llm = ChatOpenAI(model="gpt-3.5-turbo")
result = llm.invoke(prompt)
        """, language="python")
        
        st.markdown("---")
        st.subheader("Method 3: Batch Generation")
        
        st.code("""
# Generate 1000 examples efficiently
import asyncio

async def generate_batch(batch_size=10):
    tasks = []
    for i in range(batch_size):
        task = llm.ainvoke(f"Generate a customer complaint #{i}")
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

# Run
all_data = []
for batch in range(100):  # 100 batches of 10 = 1000 examples
    batch_results = await generate_batch(10)
    all_data.extend(batch_results)
    
    # Rate limiting
    await asyncio.sleep(1)
        """, language="python")

    # TAB 4: Traditional Methods
    with tabs[3]:
        st.header("🔧 Traditional Synthetic Data Methods")
        
        st.subheader("Faker Library (Python)")
        
        st.code("""
from faker import Faker
import pandas as pd

fake = Faker()

# Generate 1000 fake customer profiles
data = []
for _ in range(1000):
    data.append({
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'address': fake.address(),
        'company': fake.company(),
        'job': fake.job(),
        'credit_card': fake.credit_card_number(),
        'ssn': fake.ssn(),
        'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=80)
    })

df = pd.DataFrame(data)
df.to_csv('synthetic_customers.csv', index=False)
        """, language="python")
        
        st.markdown("---")
        st.subheader("Data Augmentation (NLP)")
        
        st.code("""
import nlpaug.augmenter.word as naw

# Original text
text = "The product quality is terrible and shipping was delayed."

# Synonym replacement
aug_synonym = naw.SynonymAug(aug_src='wordnet')
augmented = aug_synonym.augment(text, n=3)

# Results:
# 1. "The product quality is awful and shipping was postponed."
# 2. "The merchandise quality is horrible and delivery was delayed."
# 3. "The product caliber is terrible and shipping was deferred."

# Back-translation (English -> French -> English)
aug_back = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)
augmented = aug_back.augment(text)
        """, language="python")
        
        st.markdown("---")
        st.subheader("SMOTE (Tabular Data)")
        
        st.code("""
from imblearn.over_sampling import SMOTE
import numpy as np

# Original imbalanced dataset
X = np.array([[1, 2], [2, 3], [3, 4], [10, 11]])  # Features
y = np.array([0, 0, 0, 1])  # Labels (3 class 0, 1 class 1)

# Generate synthetic minority class samples
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Now balanced: 3 class 0, 3 class 1 (synthetic)
        """, language="python")

    # TAB 5: Quality Control
    with tabs[4]:
        st.header("✅ Quality Control for Synthetic Data")
        
        st.markdown("""
        **Synthetic data is only useful if it's high quality.** Always validate before using.
        """)
        
        st.subheader("Quality Checklist")
        
        quality_checks = {
            "1. Statistical Similarity": {
                "check": "Does synthetic data match real data distribution?",
                "method": "Compare mean, std, correlations",
                "tool": "pandas.describe(), scipy.stats"
            },
            "2. Diversity": {
                "check": "Are there duplicates or repetitive patterns?",
                "method": "Check unique values, n-grams",
                "tool": "df.nunique(), set()"
            },
            "3. Validity": {
                "check": "Does data follow schema/constraints?",
                "method": "Type checking, range validation",
                "tool": "pydantic, pandera"
            },
            "4. Realism": {
                "check": "Does it 'feel' real to humans?",
                "method": "Manual review of sample",
                "tool": "Human evaluation"
            },
            "5. Utility": {
                "check": "Does model trained on synthetic data work on real data?",
                "method": "Train on synthetic, test on real",
                "tool": "sklearn.metrics"
            }
        }
        
        for name, info in quality_checks.items():
            with st.expander(f"**{name}**"):
                st.markdown(f"**Check**: {info['check']}")
                st.markdown(f"**Method**: {info['method']}")
                st.code(f"Tool: {info['tool']}")
        
        st.markdown("---")
        st.subheader("Example: Validation Pipeline")
        
        st.code("""
from pandera import DataFrameSchema, Column, Check

# Define expected schema
schema = DataFrameSchema({
    "age": Column(int, Check.in_range(18, 100)),
    "email": Column(str, Check.str_matches(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$')),
    "salary": Column(float, Check.greater_than(0)),
    "department": Column(str, Check.isin(['Sales', 'Engineering', 'Marketing']))
})

# Validate synthetic data
try:
    validated_df = schema.validate(synthetic_df)
    print(f"✅ {len(validated_df)} valid records")
except Exception as e:
    print(f"❌ Validation failed: {e}")
        """, language="python")

    # TAB 6: Ethics & Risks
    with tabs[5]:
        st.header("⚖️ Ethics & Risks of Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Risks")
            st.markdown("""
            **1. Amplifying Biases**
            - LLMs trained on biased data generate biased synthetic data
            - Example: Generating "CEO" always as male names
            
            **2. Lack of Real-World Complexity**
            - Synthetic data may miss edge cases
            - Models overfit to synthetic patterns
            
            **3. Privacy Leakage**
            - LLMs may memorize training data
            - Synthetic data could reveal real patterns
            
            **4. Overconfidence**
            - Models perform well on synthetic test sets
            - Fail on real-world deployment
            
            **5. Regulatory Uncertainty**
            - Is synthetic medical data HIPAA compliant?
            - Legal gray areas
            """)
        
        with col2:
            st.subheader("✅ Best Practices")
            st.markdown("""
            **1. Always Validate on Real Data**
            - Train on synthetic, test on real
            - Never skip real-world validation
            
            **2. Audit for Bias**
            - Check demographic distributions
            - Use fairness metrics
            
            **3. Combine with Real Data**
            - 80% real + 20% synthetic
            - Synthetic for rare cases only
            
            **4. Document Provenance**
            - Track how data was generated
            - Version control prompts
            
            **5. Human Review**
            - Sample 1% for manual inspection
            - Catch obvious errors
            """)
        
        st.markdown("---")
        st.warning("""
        **⚠️ Important**: Synthetic data is a tool, not a replacement for real data. 
        Always validate models on real-world data before deployment.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔗 Resources
    - [Gretel.ai](https://gretel.ai/) - Synthetic data platform
    - [Mostly AI](https://mostly.ai/) - Privacy-preserving synthetic data
    - [Faker Documentation](https://faker.readthedocs.io/)
    - [SMOTE Paper](https://arxiv.org/abs/1106.1813)
    - [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
    """)
