"""
Evaluation metrics for poker agent performance.
Implements expected loss, CVaR, and per-round statistics.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


class Evaluator:
    """Evaluates agent performance using various metrics"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, cumulative_reward: float, per_round_rewards: List[float]):
        """
        Add a single match result.
        cumulative_reward: Total chips gained/lost over 20 rounds
        per_round_rewards: List of chip gains/losses for each round
        """
        self.results.append({
            'cumulative': cumulative_reward,
            'per_round': per_round_rewards
        })
    
    def compute_expected_loss(self) -> Tuple[float, float]:
        """
        Compute expected cumulative loss (negative of reward).
        Returns (mean_loss, std_loss)
        """
        if not self.results:
            return (0.0, 0.0)
        
        cumulative_rewards = [r['cumulative'] for r in self.results]
        losses = [-r for r in cumulative_rewards]
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        return (mean_loss, std_loss)
    
    def compute_per_round_statistics(self) -> Dict[str, float]:
        """
        Compute per-round return statistics.
        Returns dictionary with mean, std, min, max
        """
        if not self.results:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        all_round_rewards = []
        for result in self.results:
            all_round_rewards.extend(result['per_round'])
        
        return {
            'mean': np.mean(all_round_rewards),
            'std': np.std(all_round_rewards),
            'min': np.min(all_round_rewards),
            'max': np.max(all_round_rewards)
        }
    
    def compute_cvar(self, alpha: float = 0.05) -> float:
        """
        Compute Conditional Value at Risk (CVaR) at level alpha.
        CVaR is the expected loss given that loss exceeds the alpha-quantile.
        Lower is better.
        """
        if not self.results:
            return 0.0
        
        cumulative_rewards = [r['cumulative'] for r in self.results]
        losses = [-r for r in cumulative_rewards]
        
        # Sort losses in descending order
        sorted_losses = sorted(losses, reverse=True)
        
        # Find alpha-quantile
        n = len(sorted_losses)
        quantile_idx = int(np.ceil(alpha * n))
        
        if quantile_idx == 0:
            quantile_idx = 1
        
        # CVaR is the mean of losses above the quantile
        tail_losses = sorted_losses[:quantile_idx]
        
        if len(tail_losses) == 0:
            return 0.0
        
        return np.mean(tail_losses)
    
    def compute_win_rate(self) -> float:
        """
        Compute percentage of matches with non-negative cumulative reward.
        """
        if not self.results:
            return 0.0
        
        wins = sum(1 for r in self.results if r['cumulative'] >= 0)
        return wins / len(self.results)
    
    def compute_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for cumulative reward.
        Returns (lower_bound, upper_bound)
        """
        if not self.results:
            return (0.0, 0.0)
        
        cumulative_rewards = [r['cumulative'] for r in self.results]
        
        mean = np.mean(cumulative_rewards)
        std = np.std(cumulative_rewards)
        n = len(cumulative_rewards)
        
        # Use t-distribution for small samples
        if n < 30:
            t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
            margin = t_critical * std / np.sqrt(n)
        else:
            # Use normal distribution for large samples
            z_critical = stats.norm.ppf((1 + confidence) / 2)
            margin = z_critical * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of all metrics.
        """
        expected_loss, loss_std = self.compute_expected_loss()
        per_round_stats = self.compute_per_round_statistics()
        cvar_05 = self.compute_cvar(alpha=0.05)
        cvar_10 = self.compute_cvar(alpha=0.10)
        win_rate = self.compute_win_rate()
        ci_lower, ci_upper = self.compute_confidence_interval()
        
        return {
            'num_matches': len(self.results),
            'expected_loss': expected_loss,
            'loss_std': loss_std,
            'per_round_mean': per_round_stats['mean'],
            'per_round_std': per_round_stats['std'],
            'per_round_min': per_round_stats['min'],
            'per_round_max': per_round_stats['max'],
            'cvar_05': cvar_05,
            'cvar_10': cvar_10,
            'win_rate': win_rate,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }
    
    def print_summary(self):
        """Print formatted summary of evaluation metrics"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Number of matches: {summary['num_matches']}")
        print(f"\nCumulative Reward Metrics:")
        print(f"  Expected Loss: {summary['expected_loss']:.2f} ± {summary['loss_std']:.2f}")
        print(f"  95% Confidence Interval: [{summary['ci_95_lower']:.2f}, {summary['ci_95_upper']:.2f}]")
        print(f"  Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"\nRisk Metrics:")
        print(f"  CVaR (5%): {summary['cvar_05']:.2f}")
        print(f"  CVaR (10%): {summary['cvar_10']:.2f}")
        print(f"\nPer-Round Statistics:")
        print(f"  Mean: {summary['per_round_mean']:.2f}")
        print(f"  Std: {summary['per_round_std']:.2f}")
        print(f"  Min: {summary['per_round_min']:.2f}")
        print(f"  Max: {summary['per_round_max']:.2f}")
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all stored results"""
        self.results = []

