# Algorithmic Strategy for Minimizing Loss in Texas Hold'em over 20 Rounds

This project implements a POMDP (Partially Observable Markov Decision Process) based agent for playing Texas Hold'em poker with the goal of minimizing expected losses over a 20-round horizon.

## Overview

The agent uses:
- **POMDP Modeling**: Maintains belief distributions over opponent's private cards and behavioral tendencies
- **Bayesian Belief Updates**: Updates beliefs using Bayes' rule based on opponent actions
- **Monte Carlo Rollouts**: Estimates action values through simulation
- **Risk-Averse Optimization**: Focuses on minimizing losses rather than maximizing wins

## Project Structure

- `poker_game.py`: Core poker game environment with card dealing, betting, and hand evaluation
- `pomdp_agent.py`: POMDP agent implementation with belief tracking and Monte Carlo search
- `evaluation.py`: Evaluation metrics (expected loss, CVaR, per-round statistics)
- `main.py`: Main experiment script for running matches and evaluations

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a single experiment with default settings:
```bash
python main.py
```

### Custom Experiments

Run with custom parameters:
```bash
python main.py --matches 100 --rollouts 50 --opponent mixed
```

Arguments:
- `--matches`: Number of 20-round matches to run (default: 100)
- `--rollouts`: Number of Monte Carlo rollouts per action (default: 50)
- `--opponent`: Opponent type - 'tight', 'loose', 'aggressive', or 'mixed' (default: 'mixed')
- `--compare`: Compare performance against all opponent types
- `--verbose`: Print detailed match information

### Compare Against All Opponent Types

```bash
python main.py --compare --matches 50
```

## Evaluation Metrics

The evaluation system computes:

1. **Expected Cumulative Loss**: Mean and standard deviation of total losses over 20 rounds
2. **Per-Round Statistics**: Mean, std, min, max of per-round returns
3. **CVaR (Conditional Value at Risk)**: Expected loss in worst-case scenarios (5% and 10% tails)
4. **Win Rate**: Percentage of matches with non-negative cumulative reward
5. **Confidence Intervals**: 95% confidence intervals for cumulative reward

## Algorithm Details

### POMDP Formulation

The game is modeled as a POMDP with:
- **State Space**: Agent cards, opponent cards (hidden), community cards, pot size, stacks, betting history, round number
- **Action Space**: Fold, Check, Call, Bet(n), Raise(n)
- **Observations**: Agent's cards, revealed community cards, pot, stacks, betting history
- **Belief State**: Probability distribution over opponent's private cards

### Belief Updates

Beliefs are updated using Bayes' rule:
```
b_{t+1}(s) = η · Ω(o_t | s, a_t) · Σ_{s'} T(s' → s | a_t) b_t(s')
```

### Monte Carlo Rollouts

For each action, the agent:
1. Samples opponent hands from belief distribution
2. Samples remaining community cards
3. Simulates hand to completion
4. Estimates expected return with future value discounting

## Opponent Models

The system includes several opponent behavior models:
- **Tight**: Only plays strong hands
- **Loose**: Plays many hands
- **Aggressive**: Frequently bets and raises
- **Mixed**: Combination of strategies

## Example Output

```
============================================================
EVALUATION SUMMARY
============================================================
Number of matches: 100

Cumulative Reward Metrics:
  Expected Loss: 45.23 ± 123.45
  95% Confidence Interval: [-68.22, 158.68]
  Win Rate: 58.0%

Risk Metrics:
  CVaR (5%): 234.56
  CVaR (10%): 189.34

Per-Round Statistics:
  Mean: 2.26
  Std: 6.17
  Min: -45.00
  Max: 78.00
============================================================
```

## Implementation Notes

- The hand evaluator uses a simplified 5-card combination approach
- Betting actions are discretized (1x, 2x, 3x pot)
- Belief updates use hand strength heuristics to estimate action likelihoods
- Future value estimation uses a conservative discounting approach

## Future Improvements

- More sophisticated hand strength evaluation
- Better opponent modeling with learning
- Improved belief update mechanisms
- Reinforcement learning integration
- More efficient Monte Carlo sampling

## Authors

Yiran Zhang, Yu Gu, Kunzhu Xie, Andrew Hadikusumo, Enze Wang

## License

This project is for educational purposes.

