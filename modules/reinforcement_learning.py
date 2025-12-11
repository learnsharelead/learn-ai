import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

def show():
    st.title("🎮 Reinforcement Learning")
    
    st.markdown("""
    Reinforcement Learning (RL) is how agents learn to make decisions by trial and error!
    """)
    
    tabs = st.tabs(["📚 Concepts", "🎯 Key Components", "🎲 Q-Learning", "🕹️ Grid World Demo", "🏆 Applications"])
    
    # TAB 1: Concepts
    with tabs[0]:
        st.header("What is Reinforcement Learning?")
        
        st.markdown("""
        Unlike Supervised Learning (learn from labelled data) or Unsupervised Learning (find patterns),
        **Reinforcement Learning** learns by:
        
        1. Taking **actions** in an **environment**
        2. Receiving **rewards** (or penalties)
        3. Learning a **policy** to maximize cumulative reward
        """)
        
        st.subheader("The RL Loop")
        
        st.graphviz_chart("""
        digraph RL {
            rankdir=LR;
            node [shape=box, style=filled];
            
            Agent [fillcolor=lightblue];
            Environment [fillcolor=lightgreen];
            
            Agent -> Environment [label="Action"];
            Environment -> Agent [label="State, Reward"];
        }
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Key Differences from Other ML
            
            | Aspect | Supervised | Reinforcement |
            |--------|------------|---------------|
            | Data | Labeled examples | Rewards from actions |
            | Feedback | Immediate, direct | Delayed, sparse |
            | Goal | Minimize error | Maximize reward |
            """)
            
        with col2:
            st.markdown("""
            ### Real-World Analogy
            
            🐕 **Training a dog:**
            - Agent = Dog
            - Environment = Your home
            - Actions = Sit, stay, fetch
            - Rewards = Treats! 🦴
            """)
    
    # TAB 2: Key Components
    with tabs[1]:
        st.header("Key RL Components")
        
        components = [
            ("🌍 Environment", "The world the agent interacts with. Defines rules and provides feedback."),
            ("🤖 Agent", "The decision-maker that learns and takes actions."),
            ("📊 State (s)", "Current situation of the agent. What it 'sees'."),
            ("⚡ Action (a)", "What the agent can do. Defined by action space."),
            ("🎁 Reward (r)", "Numerical feedback. +ve is good, -ve is bad."),
            ("📜 Policy (π)", "Strategy: mapping from states to actions."),
            ("💰 Value Function (V)", "Expected future rewards from a state."),
            ("🎯 Q-Function (Q)", "Expected future rewards for taking action a in state s."),
        ]
        
        for emoji_name, desc in components:
            with st.expander(emoji_name):
                st.write(desc)
        
        st.subheader("Exploration vs Exploitation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Exploration
            - Try new, unknown actions
            - Might find better strategies
            - Risk: Could be suboptimal
            """)
            
        with col2:
            st.markdown("""
            ### 💪 Exploitation
            - Use known best actions
            - Maximize immediate reward
            - Risk: Miss better options
            """)
        
        st.info("""
        **Epsilon-Greedy Strategy (ε-greedy):**
        - With probability ε: Explore (random action)
        - With probability 1-ε: Exploit (best known action)
        - Common: Start with ε=1.0, decay over time
        """)
    
    # TAB 3: Q-Learning
    with tabs[2]:
        st.header("Q-Learning Algorithm")
        
        st.markdown("""
        **Q-Learning** is a model-free RL algorithm that learns the value of actions.
        
        ### The Q-Table
        
        A table storing Q-values for every (state, action) pair.
        """)
        
        # Display sample Q-table
        q_table = np.array([
            [0.0, 0.5, 0.2, 0.1],
            [0.3, 0.0, 0.8, 0.1],
            [0.1, 0.2, 0.0, 0.9],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        import pandas as pd
        q_df = pd.DataFrame(q_table, 
            index=["State 0", "State 1", "State 2", "State 3 (Goal)"],
            columns=["Left", "Right", "Up", "Down"])
        
        st.dataframe(q_df.style.highlight_max(axis=1))
        st.caption("Highest Q-value per row = best action for that state")
        
        st.subheader("The Update Rule")
        
        st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]")
        
        st.markdown("""
        Where:
        - **α (alpha)**: Learning rate (0-1). How much to update.
        - **γ (gamma)**: Discount factor (0-1). How much future rewards matter.
        - **r**: Immediate reward received.
        - **s'**: New state after action.
        - **max Q(s', a')**: Best possible future value.
        """)
        
        st.subheader("Algorithm Steps")
        
        st.code("""
1. Initialize Q-table with zeros
2. For each episode:
   a. Reset environment, observe initial state s
   b. While not done:
      - Choose action a (ε-greedy)
      - Take action, observe reward r and new state s'
      - Update Q(s, a) using the formula
      - s = s'
   c. Decay ε
3. Return learned Q-table
        """, language="text")
    
    # TAB 4: Grid World Demo
    with tabs[3]:
        st.header("🕹️ Grid World: Learn by Playing!")
        
        st.markdown("""
        Watch an agent learn to navigate a 4x4 grid to reach the goal!
        
        - 🤖 = Agent
        - 🎯 = Goal (+10 reward)
        - 🔲 = Empty cell (-0.1 per step)
        - 🚫 = Obstacle (-5 reward)
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            alpha = st.slider("Learning Rate (α)", 0.1, 1.0, 0.5, 0.1)
            gamma = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.9, 0.1)
            epsilon = st.slider("Initial Epsilon (ε)", 0.1, 1.0, 1.0, 0.1)
            episodes = st.slider("Training Episodes", 100, 1000, 500, 100)
        
        with col2:
            # Grid World Environment (simplified)
            GRID_SIZE = 4
            GOAL = (3, 3)
            OBSTACLE = (1, 1)
            
            # Initialize Q-table: 16 states x 4 actions
            if st.button("🚀 Train Agent"):
                Q = np.zeros((GRID_SIZE * GRID_SIZE, 4))  # 4 actions: up, down, left, right
                actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
                
                rewards_per_episode = []
                
                progress_bar = st.progress(0)
                status = st.empty()
                
                current_eps = epsilon
                
                for ep in range(episodes):
                    state = (0, 0)  # Start position
                    total_reward = 0
                    steps = 0
                    max_steps = 50
                    
                    while state != GOAL and steps < max_steps:
                        state_idx = state[0] * GRID_SIZE + state[1]
                        
                        # Epsilon-greedy action selection
                        if np.random.random() < current_eps:
                            action = np.random.randint(4)
                        else:
                            action = np.argmax(Q[state_idx])
                        
                        # Take action
                        new_state = (
                            max(0, min(GRID_SIZE-1, state[0] + actions[action][0])),
                            max(0, min(GRID_SIZE-1, state[1] + actions[action][1]))
                        )
                        
                        # Calculate reward
                        if new_state == GOAL:
                            reward = 10
                        elif new_state == OBSTACLE:
                            reward = -5
                            new_state = state  # Can't move into obstacle
                        else:
                            reward = -0.1
                        
                        total_reward += reward
                        
                        # Q-learning update
                        new_state_idx = new_state[0] * GRID_SIZE + new_state[1]
                        Q[state_idx, action] += alpha * (
                            reward + gamma * np.max(Q[new_state_idx]) - Q[state_idx, action]
                        )
                        
                        state = new_state
                        steps += 1
                    
                    rewards_per_episode.append(total_reward)
                    current_eps *= 0.995  # Decay epsilon
                    
                    if ep % 50 == 0:
                        progress_bar.progress(ep / episodes)
                        status.text(f"Episode {ep}/{episodes} | Reward: {total_reward:.1f} | ε: {current_eps:.3f}")
                
                progress_bar.progress(1.0)
                status.success("Training Complete!")
                
                # Store results
                st.session_state.trained_Q = Q
                st.session_state.rewards_history = rewards_per_episode
                
                # Plot learning curve
                fig = go.Figure()
                
                # Smooth rewards
                window = 20
                smoothed = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
                
                fig.add_trace(go.Scatter(y=rewards_per_episode, mode='lines', name='Raw Rewards', opacity=0.3))
                fig.add_trace(go.Scatter(y=smoothed, mode='lines', name='Smoothed (20 ep)', line=dict(width=3)))
                fig.update_layout(title="Learning Curve", xaxis_title="Episode", yaxis_title="Total Reward")
                st.plotly_chart(fig, use_container_width=True)
            
            # Show learned policy
            if 'trained_Q' in st.session_state:
                st.subheader("Learned Policy")
                
                Q = st.session_state.trained_Q
                action_symbols = ['⬆️', '⬇️', '⬅️', '➡️']
                
                grid_display = []
                for i in range(GRID_SIZE):
                    row = []
                    for j in range(GRID_SIZE):
                        if (i, j) == GOAL:
                            row.append("🎯")
                        elif (i, j) == OBSTACLE:
                            row.append("🚫")
                        else:
                            state_idx = i * GRID_SIZE + j
                            best_action = np.argmax(Q[state_idx])
                            row.append(action_symbols[best_action])
                    grid_display.append(row)
                
                import pandas as pd
                policy_df = pd.DataFrame(grid_display)
                st.dataframe(policy_df, use_container_width=True)
                st.caption("Arrows show the learned optimal action for each cell")
    
    # TAB 5: Applications
    with tabs[4]:
        st.header("🏆 Real-World RL Applications")
        
        applications = [
            {
                "name": "🎮 Game Playing",
                "examples": "AlphaGo, Atari games, Dota 2 (OpenAI Five)",
                "how": "Agent learns to maximize game score through millions of self-play games."
            },
            {
                "name": "🤖 Robotics",
                "examples": "Boston Dynamics, manipulation tasks",
                "how": "Robots learn motor control through trial and error in simulation."
            },
            {
                "name": "🚗 Autonomous Vehicles",
                "examples": "Tesla Autopilot, Waymo",
                "how": "Learn driving policies from simulated and real driving data."
            },
            {
                "name": "💰 Finance",
                "examples": "Algorithmic trading, portfolio management",
                "how": "Learn trading strategies to maximize returns while managing risk."
            },
            {
                "name": "🏭 Industrial Control",
                "examples": "Google Data Center cooling",
                "how": "Reduced cooling energy by 40% using RL to optimize HVAC systems."
            },
            {
                "name": "🗣️ LLM Fine-tuning (RLHF)",
                "examples": "ChatGPT, Claude",
                "how": "Human feedback as reward signal to align model behavior."
            },
        ]
        
        for app in applications:
            with st.expander(app["name"]):
                st.markdown(f"**Examples:** {app['examples']}")
                st.markdown(f"**How it works:** {app['how']}")
        
        st.subheader("Popular RL Libraries")
        
        st.markdown("""
        | Library | Description |
        |---------|-------------|
        | **Gymnasium** (OpenAI Gym) | Standard environments for RL |
        | **Stable-Baselines3** | Reliable RL algorithms |
        | **RLlib** (Ray) | Scalable RL for production |
        | **CleanRL** | Single-file implementations for learning |
        """)
