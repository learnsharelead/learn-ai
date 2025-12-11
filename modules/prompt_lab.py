import streamlit as st
import time
import re

def show():
    st.header("🧪 Prompt Engineering Lab")
    st.markdown("""
    Master the art of **Prompt Engineering**. Experiment with System Prompts, Temperature, and Context 
    to see how AI behavior changes. Use the **Prompt Grader** to analyze your structure.
    """)
    
    col_input, col_output = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("1️⃣ Configuration")
        
        # Scenario Selector
        scenario = st.selectbox("Load Template", [
            "Custom",
            "Code Assistant",
            "Creative Writer",
            "Data Analyst",
            "Socratic Tutor"
        ])
        
        # Template Logic
        default_sys = "You are a helpful AI assistant."
        default_user = ""
        
        if scenario == "Code Assistant":
            default_sys = "You are an expert Python programmer. Return only code blocks. No explanations unless asked."
            default_user = "Write a function to reverse a linked list."
        elif scenario == "Socratic Tutor":
            default_sys = "You are a Socratic Tutor. Never give the answer directly. Ask guiding questions to help the student learn."
            default_user = "How do I calculate the area of a circle?"
            
        sys_prompt = st.text_area("System Prompt (Context & Role)", value=default_sys, height=120)
        user_prompt = st.text_area("User Prompt (Task & Input)", value=default_user, height=120)
        
        temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, help="Higher = More Random, Lower = More Deterministic")
        
        if st.button("✨ Analyze & Run", type="primary", use_container_width=True):
            st.session_state.prompt_run = True
            st.session_state.sys_prompt_val = sys_prompt
            st.session_state.user_prompt_val = user_prompt
    
    with col_output:
        st.subheader("2️⃣ Analysis & Output")
        
        if st.session_state.get("prompt_run"):
            # --- PROMPT SCORING LOGIC ---
            score = 0
            feedback = []
            
            # 1. Role Check
            if re.search(r"you are|act as|role", sys_prompt, re.IGNORECASE):
                score += 25
                feedback.append("✅ **Role Defined:** Good job setting a persona.")
            else:
                feedback.append("❌ **Missing Role:** Try 'Act as...' or 'You are...'.")
                
            # 2. Constraints Check
            if re.search(r"only|never|limit|format|json|list", sys_prompt + user_prompt, re.IGNORECASE):
                score += 25
                feedback.append("✅ **Constraints Found:** Specific output rules detected.")
            else:
                feedback.append("⚠️ **No Constraints:** AI might ramble. Try adding 'Return only JSON'.")
                
            # 3. Context/Task Check
            if len(user_prompt) > 10:
                score += 25
                feedback.append("✅ **Clear Task:** User instruction seems sufficient.")
            else:
                feedback.append("❌ **Short Task:** Elaborate on what you want.")
                
            # 4. Context Injection (Advanced)
            if "Example:" in sys_prompt or "Example:" in user_prompt:
                score += 25
                feedback.append("✅ **Few-Shot Prompting:** Examples provided! Excellent.")
            else:
                feedback.append("💡 **Tip:** Add an 'Example:' (Few-Shot) to drastically improve quality.")
            
            # Display Score
            st.markdown(f"**Prompt Strength Score:**")
            st.progress(score / 100)
            
            with st.expander("📝 Detailed Feedback", expanded=True):
                for item in feedback:
                    st.markdown(item)
            
            st.markdown("---")
            st.markdown("**🤖 Simulated Output:**")
            
            # --- SIMULATED GENERATION ---
            # In a real app, send `sys_prompt` + `user_prompt` to OpenAI/Gemini API here.
            
            # Mock Responses based on Scenario
            mock_response = "I'm ready to help! Please connect a live API Key to generate real responses.\n\n"
            
            if "python" in sys_prompt.lower() or "linked list" in user_prompt.lower():
                mock_response = """```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```"""
            elif "socratic" in sys_prompt.lower():
                mock_response = "That's a great question! What two values defines a circle's size? one starts with 'r'..."
            
            with st.chat_message("assistant"):
                st.markdown(mock_response)
                
        else:
            st.info("👈 Configure your prompt settings to see the magic.")
            st.markdown("### 📚 Prompting Cheat Sheet")
            st.markdown("""
            1. **C**ontext: Who is the AI? (System Prompt)
            2. **O**bjective: What is the specific task?
            3. **S**tyle: How should it sound?
            4. **T**one: Formal, casual, pirate?
            5. **A**udience: Who is reading this?
            6. **R**esponse: What format? (JSON, Markdown)
            """)
