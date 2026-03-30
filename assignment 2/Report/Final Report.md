# Assignment 2: Bounded Knapsack Problem

## Part 2: Training the Agent

### Results and Comparison

The results show that applying invalid action masking significantly improves the performance of the PPO agent. The standard PPO achieves a relatively low mean reward with high variance, indicating unstable learning across different seeds. In contrast, the Masked PPO achieves a much higher mean reward with lower variance, demonstrating more stable and efficient learning.

This improvement occurs because masking prevents the agent from selecting invalid actions, effectively reducing the action space and guiding the learning process toward feasible solutions.

Interestingly, tuning the Masked PPO did not lead to further improvement. In fact, the tuned version performed slightly worse and showed higher variance compared to the base masked model. This suggests that the default masked configuration was already well-suited for this problem under the given computational constraints.

### Figure: PPO vs Masked PPO

![Comparison](../figures/part3_comparison.png)

This figure compares standard PPO with Masked PPO variants. The results clearly show that Masked PPO consistently outperforms standard PPO, confirming the effectiveness of invalid action masking.