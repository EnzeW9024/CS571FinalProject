"""
Main experiment script for evaluating POMDP agent in 20-round Texas Hold'em matches.
"""

import random
import numpy as np
from typing import List, Tuple
from poker_game import PokerGame, GameState, Action
from pomdp_agent import POMDPAgent, BeliefState, OpponentModel
from evaluation import Evaluator


class MatchRunner:
    """Runs a 20-round match between agent and opponent"""
    
    def __init__(self, initial_stack: int = 1000, small_blind: int = 10, big_blind: int = 20):
        self.game = PokerGame(initial_stack, small_blind, big_blind)
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
    
    def run_match(self, agent: POMDPAgent, opponent_model: OpponentModel, 
                  num_rounds: int = 20, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Run a complete match of num_rounds hands.
        Returns (cumulative_reward, per_round_rewards)
        """
        agent_stack = self.initial_stack
        opponent_stack = self.initial_stack
        cumulative_reward = 0.0
        per_round_rewards = []
        
        dealer = 0  # 0 = agent is dealer, 1 = opponent is dealer
        
        for round_num in range(1, num_rounds + 1):
            if verbose:
                print(f"\n--- Round {round_num} ---")
                print(f"Agent stack: {agent_stack}, Opponent stack: {opponent_stack}")
            
            # Start new hand
            state = self.game.start_new_hand(agent_stack, opponent_stack, round_num, dealer)
            
            # Initialize belief state
            # Create deck excluding agent's hole cards
            from poker_game import Deck, Card
            full_deck = [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]
            remaining_deck = [c for c in full_deck if not any(
                c.rank == h.rank and c.suit == h.suit for h in state.hole_cards_agent
            )]
            belief = BeliefState(remaining_deck, state.hole_cards_agent)
            
            # Play hand
            round_reward = self._play_hand(state, agent, opponent_model, belief, 
                                          num_rounds - round_num, verbose)
            
            # Update stacks
            agent_stack = state.agent_stack
            opponent_stack = state.opponent_stack
            
            cumulative_reward += round_reward
            per_round_rewards.append(round_reward)
            
            if verbose:
                print(f"Round reward: {round_reward:.2f}, Cumulative: {cumulative_reward:.2f}")
            
            # Switch dealer
            dealer = 1 - dealer
            
            # Check if someone is out of chips
            if agent_stack <= 0 or opponent_stack <= 0:
                if verbose:
                    print(f"Match ended early: Agent stack={agent_stack}, Opponent stack={opponent_stack}")
                break
        
        return (cumulative_reward, per_round_rewards)
    
    def _play_hand(self, state: GameState, agent: POMDPAgent, opponent_model: OpponentModel,
                   belief: BeliefState, remaining_rounds: int, verbose: bool = False) -> float:
        """Play a single hand to completion"""
        # Record initial stacks
        initial_agent_stack = state.agent_stack
        initial_opponent_stack = state.opponent_stack
        
        max_iterations = 100
        iteration = 0
        
        while not state.hand_over and iteration < max_iterations:
            iteration += 1
            
            # Agent's turn
            if not state.agent_acted and not state.hand_over:
                action, amount = agent.choose_action(state, belief, remaining_rounds, opponent_model)
                state = self.game.apply_action(state, action, amount, is_agent=True)
                
                if verbose:
                    print(f"Agent: {action.value}" + (f" {amount}" if amount else ""))
            
            # Opponent's turn
            if not state.opponent_acted and not state.hand_over:
                opp_action, opp_amount = opponent_model.choose_action(
                    self.game, state, state.hole_cards_opponent
                )
                state = self.game.apply_action(state, opp_action, opp_amount, is_agent=False)
                
                # Update belief based on opponent's action
                belief.update_from_action(
                    opp_action, opp_amount, state.community_cards,
                    state.pot, state.current_bet, state.betting_history
                )
                
                if verbose:
                    print(f"Opponent: {opp_action.value}" + (f" {opp_amount}" if opp_amount else ""))
        
        # Resolve hand and add winnings back to stacks
        agent_won, opponent_won = self.game.resolve_hand(state)
        state.agent_stack += agent_won
        state.opponent_stack += opponent_won
        
        # Calculate net gain for this hand
        net_gain = state.agent_stack - initial_agent_stack
        
        return net_gain


def run_experiments(num_matches: int = 100, num_rollouts: int = 100, 
                   opponent_type: str = 'mixed', verbose: bool = False):
    """
    Run experiments with POMDP agent against different opponent types.
    
    Args:
        num_matches: Number of 20-round matches to run
        num_rollouts: Number of Monte Carlo rollouts per action
        opponent_type: 'tight', 'loose', 'aggressive', or 'mixed'
        verbose: Whether to print detailed match information
    """
    print(f"\n{'='*60}")
    print(f"Running {num_matches} matches against {opponent_type} opponent")
    print(f"Monte Carlo rollouts per action: {num_rollouts}")
    print(f"{'='*60}\n")
    
    # Initialize agent and opponent
    agent = POMDPAgent(num_rollouts=num_rollouts, discount_factor=0.95, risk_penalty=0.1)
    opponent_model = OpponentModel(behavior_type=opponent_type)
    runner = MatchRunner(initial_stack=1000, small_blind=10, big_blind=20)
    evaluator = Evaluator()
    
    # Run matches
    for match_num in range(1, num_matches + 1):
        if verbose or match_num % 10 == 0:
            print(f"Running match {match_num}/{num_matches}...")
        
        cumulative_reward, per_round_rewards = runner.run_match(
            agent, opponent_model, num_rounds=20, verbose=False
        )
        
        evaluator.add_result(cumulative_reward, per_round_rewards)
    
    # Print evaluation summary
    evaluator.print_summary()
    
    return evaluator


def compare_opponents(num_matches: int = 50, num_rollouts: int = 100):
    """Compare agent performance against different opponent types"""
    opponent_types = ['tight', 'loose', 'aggressive', 'mixed']
    results = {}
    
    for opp_type in opponent_types:
        print(f"\n{'='*60}")
        print(f"Testing against {opp_type} opponent")
        print(f"{'='*60}")
        
        evaluator = run_experiments(
            num_matches=num_matches,
            num_rollouts=num_rollouts,
            opponent_type=opp_type,
            verbose=False
        )
        
        results[opp_type] = evaluator.get_summary()
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON ACROSS OPPONENT TYPES")
    print("="*60)
    print(f"{'Opponent':<15} {'Expected Loss':<15} {'Win Rate':<12} {'CVaR (5%)':<12}")
    print("-"*60)
    
    for opp_type, summary in results.items():
        print(f"{opp_type:<15} {summary['expected_loss']:>10.2f} ± {summary['loss_std']:<6.2f} "
              f"{summary['win_rate']*100:>8.1f}%    {summary['cvar_05']:>10.2f}")
    
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run POMDP poker agent experiments')
    parser.add_argument('--matches', type=int, default=100, help='Number of matches to run')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of Monte Carlo rollouts')
    parser.add_argument('--opponent', type=str, default='mixed', 
                       choices=['tight', 'loose', 'aggressive', 'mixed'],
                       help='Opponent behavior type')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare performance against all opponent types')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed match information')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_opponents(num_matches=args.matches, num_rollouts=args.rollouts)
    else:
        run_experiments(
            num_matches=args.matches,
            num_rollouts=args.rollouts,
            opponent_type=args.opponent,
            verbose=args.verbose
        )

