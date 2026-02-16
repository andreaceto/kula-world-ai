# 🧠 Kula-World RL + LLM Integration

Curriculum-based Reinforcement Learning and LLM-Augmented Agents in a Custom Grid Environment.

This project investigates the integration of classical reinforcement learning and Large Language Models (LLMs) in a structured Kula-World-inspired grid environment. We compare:

* ✅ Curriculum-trained MaskablePPO baseline
* 🤖 LLM-as-Policy agent (DeepSeek-based)
* 🎯 LLM-based reward shaping

The results show that curriculum RL with action masking outperforms both direct LLM control and naive LLM reward shaping at higher difficulty levels.

---

# 📦 Project Structure

```
.
├── src/
│   ├── envs/
│   │   └── kula_env.py
│   ├── agents/
│   │   ├── train_baseline.py
│   │   ├── test_baseline.py
│   │   ├── action_mask.py
│   │   ├── llm_policy.py
│   │   ├── llm_reward.py
│   │   ├── test_llm_policy.py
│   │   └── test_llm_reward.py
│
├── logs/
├── models/
├── figures/
├── report/
└── README.md
```

---

# 🎮 Environment Overview

* Grid-based environment (one-hot encoded grid)
* Objective: collect key → reach exit
* Hazards: void, spikes
* Discrete action space (8 actions: move + jump)
* Action masking prevents invalid moves
* Difficulty levels: L0–L7 (increasing layout complexity)

---

# 🏋️ Baseline: Curriculum MaskablePPO

We train a MaskablePPO agent with:

* MultiInputPolicy (dict observation)
* Action masking
* Curriculum learning
* Mixed difficulty sampling from L2 onward
* Reward normalization (VecNormalize)

### Training

```bash
python -m src.agents.train_baseline
```

---

# 🤖 LLM Variants

## 1️⃣ LLM-as-Policy

* Model: `deepseek-chat`
* Deterministic decoding (`temperature=0`)
* 5×5 local observation patch
* Structured prompt with allowed actions
* Strict output format: `ACTION=<0-7>`
* Heuristic fallback if parsing fails
* In-memory caching

### Run evaluation:

```bash
python -m src.agents.test_llm_policy --episodes 10
```

---

## 2️⃣ LLM-based Reward Shaping

* Scalar reward in [-1, 1]
* Structured transition summary
* Strict output format: `REWARD=<float>`
* Used during PPO training
* Cached for efficiency

### Evaluate trained model:

```bash
python -m src.agents.test_llm_reward --episodes 10
```

---

# 📊 Results Summary

Evaluation across L0–L7 (10 episodes per level):

| Agent        | Low Difficulty | High Difficulty | L7 Success |
| ------------ | -------------- | --------------- | ---------- |
| PPO Baseline | Near perfect   | Gradual drop    | 40%        |
| LLM Policy   | Competitive    | Fails at L6–L7  | 10%        |
| LLM Reward   | Worse overall  | Collapses       | 0%         |

### Key Observations

* Curriculum learning stabilizes training.
* Action masking eliminates hazardous deaths.
* LLM-policy struggles with long-horizon planning.
* LLM-reward shaping destabilizes PPO at higher difficulty.
* LLM inference latency (60–230 ms/step) is significantly higher than RL forward pass.
* API cost is a major practical limitation for LLM-based approaches.

---

# ⚠️ Limitations

* Only 10 evaluation episodes per difficulty.
* LLM-policy uses limited local context (5×5 patch).
* No memory across timesteps.
* LLM-reward may introduce noisy shaping.
* External API cost and latency constrain scalability.

---

# 🚀 Future Work

* Hybrid RL + LLM hierarchical control
* Larger context or memory for LLM-policy
* More principled reward integration
* Local LLM deployment to reduce latency
* Larger evaluation sample size

---

# 🛠 Installation

```bash
pip install -r requirements.txt
```

You will need:

* `stable-baselines3`
* `sb3-contrib`
* `gymnasium`
* `pygame`
* `openai`
* `python-dotenv`
* `numpy`

For LLM agents:

Create `.env` file:

```
DEEPSEEK_API_KEY=your_api_key_here
```
