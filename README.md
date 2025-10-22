# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by the policy at state s
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:
### POLICY 1:
<img width="531" height="172" alt="21rl" src="https://github.com/user-attachments/assets/4d6cd5df-2315-4f13-9c2c-993e8aadb055" />

<img width="608" height="121" alt="22rl" src="https://github.com/user-attachments/assets/b0bba76f-0b76-4f87-821d-3caeea7806dd" />

### POLICY 2:
<img width="531" height="172" alt="21rl" src="https://github.com/user-attachments/assets/e0e6a7d8-e9ce-490d-bae0-68c6ca73d365" />

<img width="602" height="118" alt="24rl" src="https://github.com/user-attachments/assets/e6e8e438-ba5f-4053-8efb-453ae01ba85e" />

### COMPARISON:
<img width="527" height="186" alt="25rl" src="https://github.com/user-attachments/assets/327825dc-851b-41d7-8997-6ee21f0d21ab" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
