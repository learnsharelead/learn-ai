import streamlit as st
import json
from datetime import datetime

# Quiz data for all modules
QUIZZES = {
    "AI Fundamentals": [
        {
            "question": "What is Machine Learning?",
            "options": [
                "A) Programming explicit rules for every scenario",
                "B) Algorithms that learn patterns from data",
                "C) A type of robot",
                "D) A programming language"
            ],
            "correct": 1,
            "explanation": "ML algorithms learn from data instead of being explicitly programmed with rules."
        },
        {
            "question": "Which type of learning uses labeled data?",
            "options": [
                "A) Unsupervised Learning",
                "B) Reinforcement Learning",
                "C) Supervised Learning",
                "D) Transfer Learning"
            ],
            "correct": 2,
            "explanation": "Supervised learning uses labeled data (input-output pairs) to train models."
        },
        {
            "question": "What did the 2017 paper 'Attention Is All You Need' introduce?",
            "options": [
                "A) Convolutional Neural Networks",
                "B) Recurrent Neural Networks",
                "C) The Transformer Architecture",
                "D) Support Vector Machines"
            ],
            "correct": 2,
            "explanation": "This paper introduced the Transformer, the foundation for GPT, BERT, and modern LLMs."
        },
        {
            "question": "What is Deep Learning?",
            "options": [
                "A) Learning very deeply about a subject",
                "B) Neural networks with multiple layers",
                "C) Learning without a teacher",
                "D) A type of database"
            ],
            "correct": 1,
            "explanation": "Deep Learning uses neural networks with many layers (deep) to solve complex problems."
        },
        {
            "question": "Which company created ChatGPT?",
            "options": [
                "A) Google",
                "B) Meta",
                "C) OpenAI",
                "D) Microsoft"
            ],
            "correct": 2,
            "explanation": "ChatGPT was created by OpenAI (with Microsoft as a major investor)."
        }
    ],
    "Supervised Learning": [
        {
            "question": "Linear Regression is used to predict:",
            "options": [
                "A) Categories/Classes",
                "B) Continuous numerical values",
                "C) Clusters",
                "D) Images"
            ],
            "correct": 1,
            "explanation": "Linear Regression predicts continuous values (like house prices, temperatures)."
        },
        {
            "question": "What does the 'slope' in y = mx + c represent?",
            "options": [
                "A) The starting point",
                "B) How much y changes for each unit of x",
                "C) The error term",
                "D) The number of data points"
            ],
            "correct": 1,
            "explanation": "The slope (m) tells us how much y changes when x increases by 1."
        },
        {
            "question": "Logistic Regression is used for:",
            "options": [
                "A) Regression problems",
                "B) Classification problems",
                "C) Clustering",
                "D) Dimensionality reduction"
            ],
            "correct": 1,
            "explanation": "Despite the name, Logistic Regression is a classification algorithm."
        },
        {
            "question": "What is Overfitting?",
            "options": [
                "A) Model is too simple",
                "B) Model memorizes training data but fails on new data",
                "C) Model trains too quickly",
                "D) Model has too few features"
            ],
            "correct": 1,
            "explanation": "Overfitting occurs when a model learns noise in training data and doesn't generalize."
        },
        {
            "question": "Which algorithm creates a tree of if-else decisions?",
            "options": [
                "A) Linear Regression",
                "B) K-Means",
                "C) Decision Tree",
                "D) PCA"
            ],
            "correct": 2,
            "explanation": "Decision Trees split data based on feature thresholds, creating a tree structure."
        }
    ],
    "Neural Networks": [
        {
            "question": "What is an Activation Function?",
            "options": [
                "A) A function that activates the computer",
                "B) A function that introduces non-linearity",
                "C) A function that loads data",
                "D) A function that saves the model"
            ],
            "correct": 1,
            "explanation": "Activation functions add non-linearity, enabling networks to learn complex patterns."
        },
        {
            "question": "What is ReLU?",
            "options": [
                "A) f(x) = 1/(1+e^-x)",
                "B) f(x) = max(0, x)",
                "C) f(x) = tanh(x)",
                "D) f(x) = x^2"
            ],
            "correct": 1,
            "explanation": "ReLU (Rectified Linear Unit) outputs x if positive, else 0. Most popular activation."
        },
        {
            "question": "Backpropagation is used to:",
            "options": [
                "A) Make predictions",
                "B) Calculate gradients and update weights",
                "C) Load data",
                "D) Visualize the network"
            ],
            "correct": 1,
            "explanation": "Backpropagation calculates how to adjust weights to minimize the loss."
        },
        {
            "question": "What is an Epoch?",
            "options": [
                "A) One prediction",
                "B) One complete pass through the training data",
                "C) One neuron activation",
                "D) One layer in the network"
            ],
            "correct": 1,
            "explanation": "An epoch is one complete pass through the entire training dataset."
        }
    ],
    "Generative AI": [
        {
            "question": "What is the core task of Large Language Models?",
            "options": [
                "A) Image classification",
                "B) Next token/word prediction",
                "C) Speech recognition",
                "D) Video generation"
            ],
            "correct": 1,
            "explanation": "LLMs fundamentally predict the next token given the previous context."
        },
        {
            "question": "What does RLHF stand for?",
            "options": [
                "A) Rapid Learning High Frequency",
                "B) Reinforcement Learning from Human Feedback",
                "C) Real-time Language Handling Framework",
                "D) Recursive Linear Hidden Function"
            ],
            "correct": 1,
            "explanation": "RLHF uses human preferences to fine-tune models to be helpful and safe."
        },
        {
            "question": "What is RAG?",
            "options": [
                "A) Random Access Generation",
                "B) Retrieval-Augmented Generation",
                "C) Rapid AI Growth",
                "D) Recursive Algorithm Generator"
            ],
            "correct": 1,
            "explanation": "RAG retrieves relevant documents and uses them to ground LLM responses."
        },
        {
            "question": "Higher Temperature in LLMs means:",
            "options": [
                "A) Faster generation",
                "B) More deterministic output",
                "C) More random/creative output",
                "D) Longer responses"
            ],
            "correct": 2,
            "explanation": "Higher temperature increases randomness in token selection."
        }
    ],
    "Model Evaluation": [
        {
            "question": "Precision measures:",
            "options": [
                "A) Of all positives, how many did we find?",
                "B) Of all positive predictions, how many were correct?",
                "C) Overall accuracy",
                "D) Model speed"
            ],
            "correct": 1,
            "explanation": "Precision = TP / (TP + FP). 'Of all I called positive, how many really were?'"
        },
        {
            "question": "When is Recall more important than Precision?",
            "options": [
                "A) Spam detection",
                "B) Cancer screening",
                "C) Movie recommendations",
                "D) Weather forecasting"
            ],
            "correct": 1,
            "explanation": "In cancer screening, missing a positive (FN) is worse than a false alarm (FP)."
        },
        {
            "question": "AUC of 0.5 means:",
            "options": [
                "A) Perfect model",
                "B) Random guessing",
                "C) Terrible model",
                "D) Model is overfitting"
            ],
            "correct": 1,
            "explanation": "AUC = 0.5 means the model is no better than random guessing."
        },
        {
            "question": "What is Cross-Validation?",
            "options": [
                "A) Validating across different datasets",
                "B) Training and testing on multiple splits of data",
                "C) Crossing out validation data",
                "D) A type of regularization"
            ],
            "correct": 1,
            "explanation": "Cross-validation splits data K ways, training on K-1 and testing on 1, repeated K times."
        }
    ]
}

def get_quiz_state_key(module_name):
    return f"quiz_{module_name.replace(' ', '_').lower()}"

def show():
    st.title("🎮 Interactive Quiz System")
    
    st.markdown("""
    Test your knowledge! Complete quizzes to earn points and track your progress.
    """)
    
    # Initialize session state for scores
    if 'quiz_scores' not in st.session_state:
        st.session_state.quiz_scores = {}
    if 'total_xp' not in st.session_state:
        st.session_state.total_xp = 0
    
    # Module selection
    available_quizzes = list(QUIZZES.keys())
    selected_module = st.selectbox("Select Module Quiz:", available_quizzes)
    
    quiz_key = get_quiz_state_key(selected_module)
    
    # Initialize quiz state if needed
    if quiz_key not in st.session_state:
        st.session_state[quiz_key] = {
            'current_question': 0,
            'answers': [],
            'submitted': False,
            'score': 0
        }
    
    quiz_state = st.session_state[quiz_key]
    questions = QUIZZES[selected_module]
    
    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total XP", st.session_state.total_xp)
    with col2:
        completed = len([s for s in st.session_state.quiz_scores.values() if s > 0])
        st.metric("✅ Quizzes Completed", f"{completed}/{len(available_quizzes)}")
    with col3:
        if selected_module in st.session_state.quiz_scores:
            st.metric(f"🏆 {selected_module} Score", f"{st.session_state.quiz_scores[selected_module]}%")
        else:
            st.metric(f"🏆 {selected_module} Score", "Not taken")
    
    st.markdown("---")
    
    if not quiz_state['submitted']:
        # Show all questions
        st.subheader(f"📝 {selected_module} Quiz ({len(questions)} Questions)")
        
        user_answers = []
        
        for i, q in enumerate(questions):
            st.markdown(f"### Question {i+1}")
            st.markdown(q['question'])
            
            answer = st.radio(
                f"Select your answer for Q{i+1}:",
                q['options'],
                key=f"{quiz_key}_q{i}",
                index=None
            )
            
            if answer:
                # Extract the option index (A=0, B=1, C=2, D=3)
                user_answers.append(ord(answer[0]) - ord('A'))
            else:
                user_answers.append(None)
            
            st.markdown("---")
        
        # Submit button
        if st.button("📤 Submit Quiz", type="primary"):
            if None in user_answers:
                st.error("⚠️ Please answer all questions before submitting!")
            else:
                # Calculate score
                correct = 0
                for i, (user_ans, q) in enumerate(zip(user_answers, questions)):
                    if user_ans == q['correct']:
                        correct += 1
                
                score_percent = int((correct / len(questions)) * 100)
                
                # Update state
                quiz_state['submitted'] = True
                quiz_state['answers'] = user_answers
                quiz_state['score'] = score_percent
                
                # Store score
                st.session_state.quiz_scores[selected_module] = score_percent
                
                # Award XP
                xp_earned = correct * 10
                st.session_state.total_xp += xp_earned
                
                st.rerun()
    
    else:
        # Show results
        score = quiz_state['score']
        user_answers = quiz_state['answers']
        
        # Score display with celebration
        if score >= 80:
            st.balloons()
            st.success(f"🎉 Excellent! You scored {score}%!")
        elif score >= 60:
            st.info(f"👍 Good job! You scored {score}%")
        else:
            st.warning(f"📚 Keep learning! You scored {score}%")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for i, (user_ans, q) in enumerate(zip(user_answers, questions)):
            is_correct = user_ans == q['correct']
            
            with st.expander(f"{'✅' if is_correct else '❌'} Question {i+1}: {q['question'][:50]}..."):
                st.markdown(f"**Question:** {q['question']}")
                st.markdown(f"**Your Answer:** {q['options'][user_ans]}")
                st.markdown(f"**Correct Answer:** {q['options'][q['correct']]}")
                
                if is_correct:
                    st.success("Correct! ✅")
                else:
                    st.error("Incorrect ❌")
                
                st.info(f"💡 **Explanation:** {q['explanation']}")
        
        # Retake button
        if st.button("🔄 Retake Quiz"):
            st.session_state[quiz_key] = {
                'current_question': 0,
                'answers': [],
                'submitted': False,
                'score': 0
            }
            st.rerun()
        
        # Certificate
        if score >= 80:
            st.markdown("---")
            st.subheader("🏆 Certificate of Completion")
            st.markdown(f"""
            <div style="border: 3px solid gold; padding: 20px; border-radius: 10px; text-align: center; background: linear-gradient(135deg, #fff9e6 0%, #fff 100%);">
                <h2 style="color: #b8860b;">🎓 Certificate of Achievement</h2>
                <p>This certifies that</p>
                <h3 style="color: #2e7d32;">AI Learner</h3>
                <p>has successfully completed the</p>
                <h3 style="color: #1565c0;">{selected_module} Module Quiz</h3>
                <p>with a score of <strong>{score}%</strong></p>
                <p style="color: #666;">Date: {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            """, unsafe_allow_html=True)
