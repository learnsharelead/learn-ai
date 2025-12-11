import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

def show():
    st.title("📊 Learning Progress Dashboard")
    
    st.markdown("Track your AI learning journey!")
    
    # Initialize progress tracking
    if 'module_progress' not in st.session_state:
        st.session_state.module_progress = {
            "AI Fundamentals": {"completed": False, "time_spent": 0, "visits": 0},
            "Data Preprocessing": {"completed": False, "time_spent": 0, "visits": 0},
            "Supervised Learning": {"completed": False, "time_spent": 0, "visits": 0},
            "Unsupervised Learning": {"completed": False, "time_spent": 0, "visits": 0},
            "Neural Networks": {"completed": False, "time_spent": 0, "visits": 0},
            "Computer Vision": {"completed": False, "time_spent": 0, "visits": 0},
            "NLP Basics": {"completed": False, "time_spent": 0, "visits": 0},
            "Generative AI": {"completed": False, "time_spent": 0, "visits": 0},
            "Model Evaluation": {"completed": False, "time_spent": 0, "visits": 0},
            "AI Ethics": {"completed": False, "time_spent": 0, "visits": 0},
        }
    
    if 'quiz_scores' not in st.session_state:
        st.session_state.quiz_scores = {}
    
    if 'total_xp' not in st.session_state:
        st.session_state.total_xp = 0
    
    if 'learning_streak' not in st.session_state:
        st.session_state.learning_streak = 1
    
    if 'badges' not in st.session_state:
        st.session_state.badges = []
    
    # Overview Stats
    st.subheader("🎯 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate stats
    total_modules = len(st.session_state.module_progress)
    quizzes_taken = len([s for s in st.session_state.quiz_scores.values() if s > 0])
    avg_score = sum(st.session_state.quiz_scores.values()) / max(len(st.session_state.quiz_scores), 1) if st.session_state.quiz_scores else 0
    
    with col1:
        st.metric("🏆 Total XP", st.session_state.total_xp, delta="+10 from last quiz")
    
    with col2:
        st.metric("📝 Quizzes Completed", f"{quizzes_taken}/5")
    
    with col3:
        st.metric("📈 Average Score", f"{avg_score:.0f}%")
    
    with col4:
        st.metric("🔥 Learning Streak", f"{st.session_state.learning_streak} days")
    
    st.markdown("---")
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Progress Chart
        st.subheader("📚 Module Progress")
        
        modules = list(st.session_state.quiz_scores.keys()) if st.session_state.quiz_scores else ["AI Fundamentals", "Supervised Learning", "Neural Networks", "Generative AI", "Model Evaluation"]
        scores = [st.session_state.quiz_scores.get(m, 0) for m in modules]
        
        if any(scores):
            fig = go.Figure(data=[
                go.Bar(
                    x=modules,
                    y=scores,
                    marker_color=['#4CAF50' if s >= 80 else '#FFC107' if s >= 60 else '#f44336' for s in scores],
                    text=[f"{s}%" for s in scores],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Quiz Scores by Module",
                yaxis_title="Score (%)",
                yaxis_range=[0, 110],
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 Complete quizzes to see your progress here!")
        
        # Learning Path Visualization
        st.subheader("🗺️ Learning Path")
        
        path_data = {
            "Module": ["Fundamentals", "Data Prep", "Supervised", "Unsupervised", "Neural Nets", "CV", "NLP", "GenAI", "Evaluation", "Ethics", "Projects"],
            "Level": [1, 2, 3, 3, 4, 5, 5, 6, 4, 5, 7],
            "Status": ["✅" if m in st.session_state.quiz_scores else "⬜" for m in 
                      ["AI Fundamentals", "Data Preprocessing", "Supervised Learning", "Unsupervised Learning", 
                       "Neural Networks", "Computer Vision", "NLP Basics", "Generative AI", "Model Evaluation", "AI Ethics", "Projects"]]
        }
        
        fig2 = go.Figure()
        
        colors = ['#4CAF50' if s == "✅" else '#E0E0E0' for s in path_data["Status"]]
        
        fig2.add_trace(go.Scatter(
            x=list(range(len(path_data["Module"]))),
            y=path_data["Level"],
            mode='markers+lines+text',
            marker=dict(size=30, color=colors, line=dict(width=2, color='white')),
            line=dict(color='#BDBDBD', width=2, dash='dot'),
            text=path_data["Status"],
            textposition='middle center',
            textfont=dict(size=16),
            hovertext=path_data["Module"]
        ))
        
        fig2.update_layout(
            title="Your Learning Journey",
            xaxis=dict(tickvals=list(range(len(path_data["Module"]))), ticktext=path_data["Module"], tickangle=45),
            yaxis=dict(title="Difficulty Level", range=[0, 8]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_right:
        # Badges
        st.subheader("🏅 Badges")
        
        # Check for new badges
        all_badges = [
            {"name": "🌟 First Quiz", "condition": quizzes_taken >= 1, "desc": "Complete your first quiz"},
            {"name": "📚 Scholar", "condition": quizzes_taken >= 3, "desc": "Complete 3 quizzes"},
            {"name": "🎓 Graduate", "condition": quizzes_taken >= 5, "desc": "Complete all quizzes"},
            {"name": "💯 Perfectionist", "condition": any(s == 100 for s in st.session_state.quiz_scores.values()), "desc": "Get 100% on a quiz"},
            {"name": "🔥 On Fire", "condition": st.session_state.learning_streak >= 3, "desc": "3-day streak"},
            {"name": "🚀 XP Hunter", "condition": st.session_state.total_xp >= 100, "desc": "Earn 100 XP"},
        ]
        
        earned_badges = [b for b in all_badges if b["condition"]]
        locked_badges = [b for b in all_badges if not b["condition"]]
        
        if earned_badges:
            for badge in earned_badges:
                st.success(f"{badge['name']}")
                st.caption(badge['desc'])
        else:
            st.info("Complete activities to earn badges!")
        
        st.markdown("---")
        st.markdown("**🔒 Locked Badges**")
        for badge in locked_badges[:3]:
            st.markdown(f"~~{badge['name']}~~ - {badge['desc']}")
        
        # XP Breakdown
        st.markdown("---")
        st.subheader("💰 XP Breakdown")
        
        xp_sources = {
            "Quiz Completions": quizzes_taken * 20,
            "High Scores (80%+)": len([s for s in st.session_state.quiz_scores.values() if s >= 80]) * 30,
            "Perfect Scores": len([s for s in st.session_state.quiz_scores.values() if s == 100]) * 50,
        }
        
        for source, xp in xp_sources.items():
            if xp > 0:
                st.markdown(f"• **{source}**: +{xp} XP")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("🎮 Take a Quiz", use_container_width=True):
            st.info("Go to the Quiz System from the sidebar!")
        
        if st.button("🔄 Reset Progress", use_container_width=True):
            if st.session_state.get('confirm_reset'):
                st.session_state.quiz_scores = {}
                st.session_state.total_xp = 0
                st.session_state.badges = []
                st.session_state.confirm_reset = False
                st.success("Progress reset!")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")
    
    # Weekly Activity Heatmap
    st.markdown("---")
    st.subheader("📅 Weekly Activity")
    
    # Mock activity data (in a real app, this would be tracked)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    
    # Generate some activity data based on current state
    import numpy as np
    np.random.seed(42)
    activity = np.random.randint(0, 5, size=(4, 7))
    if st.session_state.total_xp > 0:
        activity[-1, -1] = max(3, activity[-1, -1])  # Today has activity
    
    fig3 = go.Figure(data=go.Heatmap(
        z=activity,
        x=days,
        y=weeks,
        colorscale='Greens',
        showscale=False
    ))
    
    fig3.update_layout(
        title="Learning Activity (Modules Visited)",
        height=250
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if quizzes_taken == 0:
        st.info("🎯 **Start Here:** Take the AI Fundamentals quiz to begin tracking your progress!")
    elif avg_score < 60:
        st.warning("📚 **Tip:** Review the modules where you scored below 60% and retake the quizzes.")
    elif quizzes_taken < 5:
        remaining = 5 - quizzes_taken
        st.info(f"🚀 **Almost There:** Complete {remaining} more quiz{'es' if remaining > 1 else ''} to earn the Graduate badge!")
    else:
        st.success("🎉 **Congratulations!** You've completed all quizzes. Consider revisiting low-scoring modules.")
