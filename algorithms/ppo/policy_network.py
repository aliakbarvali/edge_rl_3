"""
algorithms/ppo/policy_network.py
 
"""

PPO_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),   
)