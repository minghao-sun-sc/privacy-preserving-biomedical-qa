import numpy as np
import matplotlib.pyplot as plt

n = 30
top_k = 7
top_p = 0.3
epsilon = 1
alpha = 1
scores = np.random.beta(5., 4., n)

@np.vectorize
def top_k_u(tau: float) -> float:
    u = 0
    for s in scores:
        if tau < s:
            u+=1
    return -np.abs(u-top_k)

taus = np.linspace(0,1,1000)

fig, ax1 = plt.subplots()
ax1.plot(taus, top_k_u(taus), color='tab:blue', linewidth=2)
ax1.set_xlabel(r'$\tau$')
ax1.set_ylabel('u', color='tab:blue')

ax2 = ax1.twinx()
ax2.plot(taus, np.exp(epsilon*top_k_u(taus)/2), color='tab:orange', linewidth=2)
ax2.set_ylabel(r'$\exp\left(\frac{\epsilon u}{2}\right)$', color='tab:orange')

# fig.tight_layout()  # Adjust layout to prevent overlap
plt.title(f"Utility and pdf used in the top-k exponential mechanism (k={top_k})")
plt.savefig('documents/figures/top-k-exp.svg')

min_score = 0.2
max_score = 0.9

@np.vectorize
def top_p_u(tau: float) -> float:
    z = 0
    u = 0
    for s in scores:
        s = max(min(s, max_score), min_score)
        p = np.exp(alpha*(s-max_score)/(max_score-min_score))
        z += p
        if tau < s:
            u += p
    return -np.abs(u-top_p*z)


fig, ax1 = plt.subplots()
ax1.plot(taus, top_p_u(taus), color='tab:blue', linewidth=2)
ax1.set_xlabel(r'$\tau$')
ax1.set_ylabel('u', color='tab:blue')

ax2 = ax1.twinx()
ax2.plot(taus, np.exp(epsilon*top_p_u(taus)/2), color='tab:orange', linewidth=2)
ax2.set_ylabel(r'$\exp\left(\frac{\epsilon u}{2}\right)$', color='tab:orange')

# fig.tight_layout()  # Adjust layout to prevent overlap
plt.title(rf"Utility and pdf used in the top-p exponential mechanism (p={top_p}, $\alpha$=1)")
plt.savefig('documents/figures/top-p-exp.svg')