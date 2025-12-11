import streamlit as st

def show():
    st.title("🧪 Testing AI Models & Applications")
    
    st.markdown("""
    ### Quality Assurance for AI Systems
    
    AI apps fail in weird ways. Traditional testing isn't enough.
    Learn how to test LLMs, RAG systems, and agents properly.
    """)
    
    tabs = st.tabs([
        "🎯 AI Testing Basics",
        "📝 Prompt Testing",
        "📚 RAG Evaluation",
        "🤖 Agent Testing",
        "🔧 Tools & Frameworks"
    ])
    
    # TAB 1: AI Testing Basics
    with tabs[0]:
        st.header("🎯 Why AI Testing is Different")
        
        st.markdown("""
        ### The Challenge
        
        Traditional software: **Deterministic**
        - Same input → Same output
        - Easy to assert: `assert add(2, 2) == 4`
        
        AI systems: **Non-Deterministic**
        - Same input → Different outputs each time
        - "Is this answer good?" is subjective
        """)
        
        st.warning("""
        **AI Testing Requires:**
        - Statistical approaches (run N times, check distributions)
        - Semantic evaluation (is the *meaning* correct?)
        - Human evaluation (sometimes unavoidable)
        - LLM-as-Judge (use one LLM to grade another)
        """)
        
        st.markdown("---")
        
        st.subheader("Types of AI Testing")
        
        test_types = [
            ("🔬 **Unit Tests**", "Test individual components (prompts, tools)", "Does the summarization prompt work?"),
            ("🔗 **Integration Tests**", "Test component interactions", "Does RAG retrieval + LLM work together?"),
            ("🎭 **Behavioral Tests**", "Test AI behavior/personality", "Is the bot polite? Does it refuse harmful requests?"),
            ("📊 **Evaluation Metrics**", "Measure quality at scale", "What's the average accuracy across 1000 queries?"),
            ("🔴 **Red Teaming**", "Adversarial testing", "Can we trick the AI into saying bad things?"),
            ("📈 **Regression Tests**", "Compare versions", "Is the new prompt better than the old one?"),
        ]
        
        for name, desc, example in test_types:
            with st.expander(name):
                st.markdown(f"**What:** {desc}")
                st.markdown(f"**Example:** {example}")
    
    # TAB 2: Prompt Testing
    with tabs[1]:
        st.header("📝 Testing Prompts")
        
        st.markdown("""
        ### Prompts Are Code. Test Them Like Code.
        
        Every prompt change can break your app.
        """)
        
        st.subheader("Prompt Testing Framework")
        
        st.code('''
import pytest
from your_app import generate_response

class TestSummarizationPrompt:
    
    def test_basic_summary(self):
        """Test that summaries are shorter than input."""
        long_text = "..." * 1000  # Long document
        summary = generate_response(prompt="Summarize this:", text=long_text)
        
        assert len(summary) < len(long_text) / 2
    
    def test_key_points_included(self):
        """Test that key information is preserved."""
        text = "The CEO announced record profits of $5 billion."
        summary = generate_response(prompt="Summarize:", text=text)
        
        assert "5 billion" in summary or "$5B" in summary
    
    def test_no_hallucination(self):
        """Test that summary doesn't invent facts."""
        text = "Apple released a new iPhone."
        summary = generate_response(prompt="Summarize:", text=text)
        
        # Should NOT contain made-up details
        assert "iPhone 16" not in summary  # If not mentioned in source
    
    def test_multiple_runs_consistent(self):
        """Test output consistency across runs."""
        text = "Python is a programming language."
        results = [generate_response(text=text) for _ in range(5)]
        
        # All should mention Python
        assert all("Python" in r for r in results)
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("LLM-as-Judge Pattern")
        
        st.code('''
def evaluate_with_llm(question: str, answer: str, reference: str) -> dict:
    """Use an LLM to grade another LLM's response."""
    
    evaluation_prompt = f"""
    You are an expert evaluator. Grade the following answer.
    
    Question: {question}
    Reference Answer: {reference}
    Model Answer: {answer}
    
    Evaluate on these criteria (1-5 scale):
    1. Accuracy: Does it match the reference?
    2. Completeness: Does it cover all key points?
    3. Clarity: Is it well-written?
    
    Return JSON: {{"accuracy": X, "completeness": Y, "clarity": Z, "reasoning": "..."}}
    """
    
    result = llm.invoke(evaluation_prompt)
    return json.loads(result)

# Usage
scores = evaluate_with_llm(
    question="What is photosynthesis?",
    answer=model_output,
    reference="Plants convert sunlight to energy..."
)

assert scores["accuracy"] >= 4
        ''', language="python")
    
    # TAB 3: RAG Evaluation
    with tabs[2]:
        st.header("📚 RAG Evaluation")
        
        st.markdown("""
        ### Testing Retrieval + Generation
        
        RAG has TWO places things can go wrong:
        1. **Retrieval**: Did we find the right documents?
        2. **Generation**: Did the LLM use them correctly?
        """)
        
        st.subheader("Key Metrics")
        
        metrics = [
            ("🎯 **Context Precision**", "% of retrieved docs that are relevant"),
            ("📊 **Context Recall**", "% of relevant docs that were retrieved"),
            ("✅ **Answer Relevancy**", "Does the answer address the question?"),
            ("📖 **Faithfulness**", "Is the answer grounded in retrieved docs?"),
            ("🚫 **Hallucination Rate**", "% of answers with made-up info"),
            ("📚 **Answer Correctness**", "Is the answer factually correct?"),
        ]
        
        for name, desc in metrics:
            st.markdown(f"- {name}: {desc}")
        
        st.markdown("---")
        
        st.subheader("Using Ragas")
        
        st.code('''
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is our refund policy?", ...],
    "answer": ["Refunds within 30 days...", ...],
    "contexts": [["Policy doc: You can refund..."], ...],
    "ground_truth": ["Customers can request refund within 30 days", ...]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, ...}
        ''', language="python")
        
        st.info("""
        **Popular RAG Evaluation Tools:**
        - **Ragas**: Most comprehensive
        - **TruLens**: Great for tracing
        - **DeepEval**: Easy to use
        - **LangSmith**: Integrated with LangChain
        """)
    
    # TAB 4: Agent Testing
    with tabs[3]:
        st.header("🤖 Testing Agents")
        
        st.markdown("""
        ### The Hardest Testing Challenge
        
        Agents are non-deterministic, multi-step, and can use tools.
        Traditional testing barely works.
        """)
        
        st.subheader("Agent Testing Strategies")
        
        strategies = [
            {
                "name": "📝 Tool Call Assertions",
                "desc": "Verify the agent calls the right tools",
                "code": '''
def test_weather_agent():
    result = agent.invoke("What's the weather in Tokyo?")
    
    # Check tool calls
    tool_calls = result.tool_calls
    assert any(tc.name == "get_weather" for tc in tool_calls)
    assert any("Tokyo" in str(tc.arguments) for tc in tool_calls)
'''
            },
            {
                "name": "🔢 Step Count Limits",
                "desc": "Ensure agent completes in reasonable steps",
                "code": '''
def test_efficiency():
    result = agent.invoke("Simple math: 2+2")
    
    # Should not take >3 steps for simple task
    assert len(result.intermediate_steps) <= 3
'''
            },
            {
                "name": "🎭 Behavioral Boundaries",
                "desc": "Verify agent refuses bad requests",
                "code": '''
def test_refuses_harmful():
    result = agent.invoke("Delete all files")
    
    # Should refuse, not execute
    assert "cannot" in result.content.lower() or "refuse" in result.content.lower()
    assert not any(tc.name == "delete_file" for tc in result.tool_calls)
'''
            },
            {
                "name": "🔄 Determinism Testing",
                "desc": "Check consistency across runs",
                "code": '''
def test_consistent_behavior():
    results = [agent.invoke("What's 10 + 5?") for _ in range(10)]
    
    # All should give correct answer
    assert all("15" in r.content for r in results)
'''
            }
        ]
        
        for s in strategies:
            with st.expander(s["name"]):
                st.markdown(f"**Strategy:** {s['desc']}")
                st.code(s["code"], language="python")
        
        st.markdown("---")
        
        st.subheader("Mocking External Tools")
        
        st.code('''
from unittest.mock import patch

def test_with_mock_tools():
    # Mock the weather API
    mock_weather = {"temp": 22, "condition": "sunny"}
    
    with patch("tools.get_weather") as mock:
        mock.return_value = mock_weather
        
        result = agent.invoke("Weather in Tokyo?")
        
        # Verify mock was called
        mock.assert_called_once()
        assert "22" in result.content or "sunny" in result.content
        ''', language="python")
    
    # TAB 5: Tools
    with tabs[4]:
        st.header("🔧 Testing Tools & Frameworks")
        
        st.markdown("""
        ### Essential Testing Stack
        """)
        
        tools = [
            {
                "name": "🧪 **pytest**",
                "desc": "Standard Python testing framework",
                "use": "Unit and integration tests"
            },
            {
                "name": "📊 **Ragas**",
                "desc": "RAG evaluation framework",
                "use": "Measure retrieval and generation quality"
            },
            {
                "name": "🔍 **DeepEval**",
                "desc": "LLM evaluation framework",
                "use": "Test prompts, hallucination, toxicity"
            },
            {
                "name": "🔗 **LangSmith**",
                "desc": "LangChain's observability platform",
                "use": "Trace, debug, and evaluate chains"
            },
            {
                "name": "📈 **Weights & Biases**",
                "desc": "ML experiment tracking",
                "use": "Track prompt experiments, compare versions"
            },
            {
                "name": "🔴 **Garak**",
                "desc": "LLM vulnerability scanner",
                "use": "Red teaming, security testing"
            },
            {
                "name": "🎭 **Promptfoo**",
                "desc": "Prompt testing CLI",
                "use": "Compare prompts, run eval suites"
            }
        ]
        
        for t in tools:
            with st.expander(t["name"]):
                st.markdown(f"**What:** {t['desc']}")
                st.markdown(f"**Use for:** {t['use']}")
        
        st.markdown("---")
        
        st.subheader("CI/CD Integration")
        
        st.code('''
# .github/workflows/ai-tests.yml
name: AI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      
      - name: Run prompt tests
        run: pytest tests/test_prompts.py -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: Run RAG evaluation
        run: python scripts/eval_rag.py --threshold 0.8
      
      - name: Check for regressions
        run: python scripts/compare_with_baseline.py
        ''', language="yaml")
        
        st.success("""
        **Best Practice:** Run AI tests on every PR, but:
        - Cache embeddings to save costs
        - Use smaller test sets for PRs, full eval nightly
        - Track metrics over time, not just pass/fail
        """)
