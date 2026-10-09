"""Reinforcement-learning stage: the Equation (11) reward, PPO, and the environment.

- ``environment``  Gymnasium ``PortfolioEnv`` over one stored market path.
- ``reward``       Equations (10), (11), (16), (17).
- ``stochastic``   Gaussian actor-critic; PPO needs a density, the policies are
                   deterministic, so the Gaussian sits on the pre-projection scores.
- ``ppo``          The clipped-surrogate trainer (Weeks 30-33).
- ``analytic``     Exact-gradient baseline on the same reward, reported for
                   comparison rather than as a competing method.
- ``diagnostics``  The six acceptance checks for a PPO run.
"""
