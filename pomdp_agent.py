"""
POMDP-based Agent for Texas Hold'em
Implements belief tracking over opponent cards and behavior, and Monte Carlo rollouts for action evaluation.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from poker_game import PokerGame, GameState, Action, Card, HandEvaluator


class BeliefState:
    """Maintains belief distribution over opponent's private cards and behavioral tendencies"""
    
    def __init__(self, deck: List[Card], agent_hole: List[Card]):
        """
        Initialize belief state.
        deck: Remaining cards in deck (excluding agent's hole cards)
        agent_hole: Agent's hole cards
        """
        self.deck = deck
        self.agent_hole = agent_hole
        self.opponent_hand_probs = {}  # Probability distribution over possible opponent hands
        self.opponent_behavior_probs = {
            'tight': 0.5,      # Probability of tight play
            'loose': 0.3,      # Probability of loose play
            'aggressive': 0.2  # Probability of aggressive play
        }
        self._initialize_hand_distribution()
    
    def _initialize_hand_distribution(self):
        """Initialize uniform distribution over all possible opponent hands"""
        from itertools import combinations
        
        possible_hands = list(combinations(self.deck, 2))
        prob = 1.0 / len(possible_hands)
        
        for hand in possible_hands:
            hand_tuple = tuple(sorted(hand, key=lambda c: (c.rank, c.suit)))
            self.opponent_hand_probs[hand_tuple] = prob
    
    def update_from_action(self, action: Action, amount: Optional[int], 
                          community_cards: List[Card], pot: int, 
                          current_bet: int, betting_history: List):
        """
        Update belief distribution using Bayes' rule based on opponent's action.
        """
        # Remove community cards from possible opponent hands
        self._remove_community_cards(community_cards)
        
        # Update hand probabilities based on action likelihood
        action_likelihood = self._compute_action_likelihood(
            action, amount, community_cards, pot, current_bet
        )
        
        # Bayesian update
        total_prob = 0.0
        for hand in self.opponent_hand_probs:
            likelihood = action_likelihood.get(hand, 0.1)  # Default small likelihood
            self.opponent_hand_probs[hand] *= likelihood
            total_prob += self.opponent_hand_probs[hand]
        
        # Normalize
        if total_prob > 0:
            for hand in self.opponent_hand_probs:
                self.opponent_hand_probs[hand] /= total_prob
        else:
            # Reset to uniform if all probabilities become too small
            self._initialize_hand_distribution()
    
    def _remove_community_cards(self, community_cards: List[Card]):
        """Remove community cards from deck and update hand probabilities"""
        if not community_cards:
            return
        
        # Remove hands that contain community cards
        hands_to_remove = []
        for hand in self.opponent_hand_probs:
            for card in hand:
                if any(c.rank == card.rank and c.suit == card.suit for c in community_cards):
                    hands_to_remove.append(hand)
                    break
        
        for hand in hands_to_remove:
            del self.opponent_hand_probs[hand]
        
        # Renormalize
        total = sum(self.opponent_hand_probs.values())
        if total > 0:
            for hand in self.opponent_hand_probs:
                self.opponent_hand_probs[hand] /= total
    
    def _compute_action_likelihood(self, action: Action, amount: Optional[int],
                                   community_cards: List[Card], pot: int, 
                                   current_bet: int) -> Dict:
        """
        Compute likelihood of action given each possible opponent hand.
        Returns dictionary mapping hand -> likelihood
        """
        likelihoods = {}
        evaluator = HandEvaluator()
        
        for hand in self.opponent_hand_probs:
            hand_list = list(hand)
            hand_strength = evaluator.evaluate_hand(hand_list, community_cards)
            hand_rank, _ = hand_strength
            
            # Estimate hand strength (0-1 scale, higher is better)
            # Invert hand_rank so higher values = stronger hands
            strength = 1.0 - (hand_rank / 10.0)
            
            # Compute action likelihood based on hand strength and action type
            if action == Action.FOLD:
                # More likely to fold with weak hands
                likelihoods[hand] = 1.0 - strength
            elif action == Action.CHECK or action == Action.CALL:
                # More likely to check/call with medium hands
                likelihoods[hand] = 1.0 - abs(strength - 0.5) * 2
            elif action == Action.BET or action == Action.RAISE:
                # More likely to bet/raise with strong hands
                likelihoods[hand] = strength
            
            # Adjust based on bet size (larger bets suggest stronger hands)
            if amount is not None and pot > 0:
                bet_ratio = amount / pot if pot > 0 else 0
                if bet_ratio > 1.0:  # Large bet/raise
                    likelihoods[hand] *= (1.0 + strength)
                elif bet_ratio < 0.5:  # Small bet
                    likelihoods[hand] *= (0.5 + strength * 0.5)
        
        return likelihoods
    
    def sample_opponent_hand(self) -> List[Card]:
        """Sample an opponent hand from the belief distribution"""
        if not self.opponent_hand_probs:
            # Fallback: sample randomly from deck
            return random.sample(self.deck, min(2, len(self.deck)))
        
        hands = list(self.opponent_hand_probs.keys())
        probs = list(self.opponent_hand_probs.values())
        
        # Normalize probabilities
        total = sum(probs)
        if total == 0:
            return random.sample(self.deck, min(2, len(self.deck)))
        
        probs = [p / total for p in probs]
        
        sampled_hand = np.random.choice(len(hands), p=probs)
        return list(hands[sampled_hand])
    
    def get_hand_probability(self, hand: Tuple[Card, Card]) -> float:
        """Get probability of a specific opponent hand"""
        hand_tuple = tuple(sorted(hand, key=lambda c: (c.rank, c.suit)))
        return self.opponent_hand_probs.get(hand_tuple, 0.0)


class OpponentModel:
    """Models opponent's behavior for simulation"""
    
    def __init__(self, behavior_type: str = 'mixed'):
        """
        behavior_type: 'tight', 'loose', 'aggressive', or 'mixed'
        """
        self.behavior_type = behavior_type
    
    def choose_action(self, game: PokerGame, state: GameState, 
                     opponent_hole: List[Card]) -> Tuple[Action, Optional[int]]:
        """
        Choose action for opponent based on hand strength and behavior model.
        """
        evaluator = HandEvaluator()
        hand_strength = evaluator.evaluate_hand(opponent_hole, state.community_cards)
        hand_rank, tie_breaker = hand_strength
        
        # Convert hand rank to strength (0-1, higher is better)
        strength = 1.0 - (hand_rank / 10.0)
        
        legal_actions = game.get_legal_actions(state, is_agent=False)
        
        if self.behavior_type == 'tight':
            return self._tight_strategy(legal_actions, strength, state)
        elif self.behavior_type == 'loose':
            return self._loose_strategy(legal_actions, strength, state)
        elif self.behavior_type == 'aggressive':
            return self._aggressive_strategy(legal_actions, strength, state)
        else:  # mixed
            return self._mixed_strategy(legal_actions, strength, state)
    
    def _tight_strategy(self, legal_actions: List, strength: float, state: GameState) -> Tuple[Action, Optional[int]]:
        """Tight: only play strong hands"""
        if strength < 0.3:
            # Weak hand, fold or check
            for action, amount in legal_actions:
                if action == Action.FOLD:
                    return (action, amount)
                if action == Action.CHECK:
                    return (action, amount)
        elif strength < 0.6:
            # Medium hand, check or call
            for action, amount in legal_actions:
                if action == Action.CHECK:
                    return (action, amount)
                if action == Action.CALL:
                    return (action, amount)
        else:
            # Strong hand, bet or raise
            for action, amount in legal_actions:
                if action == Action.BET or action == Action.RAISE:
                    return (action, amount)
            for action, amount in legal_actions:
                if action == Action.CALL:
                    return (action, amount)
        
        # Default: check or fold
        for action, amount in legal_actions:
            if action == Action.CHECK:
                return (action, amount)
        return legal_actions[0]
    
    def _loose_strategy(self, legal_actions: List, strength: float, state: GameState) -> Tuple[Action, Optional[int]]:
        """Loose: play many hands"""
        if strength < 0.2:
            # Very weak, might still call
            for action, amount in legal_actions:
                if action == Action.CHECK:
                    return (action, amount)
                if action == Action.CALL and state.current_bet < state.pot * 0.5:
                    return (action, amount)
        elif strength < 0.5:
            # Medium, call or small bet
            for action, amount in legal_actions:
                if action == Action.CALL:
                    return (action, amount)
                if action == Action.BET and amount <= state.pot:
                    return (action, amount)
        else:
            # Strong, bet or raise
            for action, amount in legal_actions:
                if action == Action.BET or action == Action.RAISE:
                    return (action, amount)
        
        # Default
        for action, amount in legal_actions:
            if action == Action.CHECK:
                return (action, amount)
        return legal_actions[0]
    
    def _aggressive_strategy(self, legal_actions: List, strength: float, state: GameState) -> Tuple[Action, Optional[int]]:
        """Aggressive: bet and raise frequently"""
        if strength < 0.2:
            # Weak but aggressive, might bluff
            if random.random() < 0.3:  # 30% chance to bluff
                for action, amount in legal_actions:
                    if action == Action.BET:
                        return (action, amount)
            else:
                for action, amount in legal_actions:
                    if action == Action.FOLD:
                        return (action, amount)
        elif strength < 0.6:
            # Medium, bet or raise
            for action, amount in legal_actions:
                if action == Action.BET or action == Action.RAISE:
                    return (action, amount)
        else:
            # Strong, large bet or raise
            for action, amount in legal_actions:
                if action == Action.RAISE:
                    return (action, amount)
                if action == Action.BET:
                    return (action, amount)
        
        # Default
        for action, amount in legal_actions:
            if action == Action.CALL:
                return (action, amount)
        return legal_actions[0]
    
    def _mixed_strategy(self, legal_actions: List, strength: float, state: GameState) -> Tuple[Action, Optional[int]]:
        """Mixed: combination of strategies"""
        rand = random.random()
        if rand < 0.3:
            return self._tight_strategy(legal_actions, strength, state)
        elif rand < 0.6:
            return self._loose_strategy(legal_actions, strength, state)
        else:
            return self._aggressive_strategy(legal_actions, strength, state)


class POMDPAgent:
    """POMDP-based agent using Monte Carlo rollouts for decision making"""
    
    def __init__(self, num_rollouts: int = 100, discount_factor: float = 0.95):
        """
        num_rollouts: Number of Monte Carlo simulations per action
        discount_factor: Discount factor for future rewards
        """
        self.num_rollouts = num_rollouts
        self.discount_factor = discount_factor
        self.game = PokerGame()
        self.evaluator = HandEvaluator()
        self.opponent_model = OpponentModel(behavior_type='mixed')
    
    def choose_action(self, state: GameState, belief: BeliefState, 
                     remaining_rounds: int) -> Tuple[Action, Optional[int]]:
        """
        Choose action using Monte Carlo rollouts.
        Returns (action, amount)
        """
        legal_actions = self.game.get_legal_actions(state, is_agent=True)
        
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Evaluate each action using Monte Carlo rollouts
        action_values = {}
        
        for action, amount in legal_actions:
            q_value = self._monte_carlo_rollout(state, action, amount, belief, remaining_rounds)
            action_values[(action, amount)] = q_value
        
        # Select action with highest Q-value
        best_action = max(action_values.items(), key=lambda x: x[1])
        return best_action[0]
    
    def _monte_carlo_rollout(self, state: GameState, action: Action, amount: Optional[int],
                            belief: BeliefState, remaining_rounds: int) -> float:
        """
        Perform Monte Carlo rollout to estimate Q-value of action.
        Returns expected return.
        """
        total_return = 0.0
        
        for _ in range(self.num_rollouts):
            # Sample opponent hand from belief
            opponent_hand = belief.sample_opponent_hand()
            
            # Create a copy of state for simulation
            sim_state = self._copy_state(state)
            sim_state.hole_cards_opponent = opponent_hand
            
            # Apply agent's action
            sim_state = self.game.apply_action(sim_state, action, amount, is_agent=True)
            
            # Simulate to end of hand
            reward = self._simulate_hand(sim_state, opponent_hand, remaining_rounds)
            
            total_return += reward
        
        return total_return / self.num_rollouts
    
    def _simulate_hand(self, state: GameState, opponent_hand: List[Card], 
                      remaining_rounds: int) -> float:
        """
        Simulate hand to completion and return reward.
        """
        # If hand is over, resolve it
        if state.hand_over:
            agent_won, opponent_won = self.game.resolve_hand(state)
            return agent_won - (state.pot - agent_won)  # Net chips gained
        
        # Continue simulation
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while not state.hand_over and iteration < max_iterations:
            iteration += 1
            
            # Opponent's turn
            if not state.opponent_acted:
                opp_action, opp_amount = self.opponent_model.choose_action(
                    self.game, state, opponent_hand
                )
                state = self.game.apply_action(state, opp_action, opp_amount, is_agent=False)
            
            # Agent's turn (use simple heuristic for simulation)
            if not state.agent_acted and not state.hand_over:
                agent_action, agent_amount = self._heuristic_action(state)
                state = self.game.apply_action(state, agent_action, agent_amount, is_agent=True)
        
        # Resolve hand
        agent_won, opponent_won = self.game.resolve_hand(state)
        initial_stack = state.agent_stack + agent_won
        net_gain = agent_won - (state.pot - agent_won)
        
        # Add future value estimate (discounted)
        future_value = self._estimate_future_value(
            state.agent_stack + net_gain, remaining_rounds - 1
        )
        
        return net_gain + self.discount_factor * future_value
    
    def _heuristic_action(self, state: GameState) -> Tuple[Action, Optional[int]]:
        """Simple heuristic for agent actions during simulation"""
        hand_strength = self.evaluator.evaluate_hand(
            state.hole_cards_agent, state.community_cards
        )
        hand_rank, _ = hand_strength
        strength = 1.0 - (hand_rank / 10.0)
        
        legal_actions = self.game.get_legal_actions(state, is_agent=True)
        
        if strength < 0.3:
            # Weak hand
            for action, amount in legal_actions:
                if action == Action.CHECK:
                    return (action, amount)
                if action == Action.FOLD:
                    return (action, amount)
        elif strength < 0.6:
            # Medium hand
            for action, amount in legal_actions:
                if action == Action.CHECK:
                    return (action, amount)
                if action == Action.CALL:
                    return (action, amount)
        else:
            # Strong hand
            for action, amount in legal_actions:
                if action == Action.BET or action == Action.RAISE:
                    return (action, amount)
        
        return legal_actions[0]
    
    def _estimate_future_value(self, current_stack: int, remaining_rounds: int) -> float:
        """
        Estimate expected value of remaining rounds.
        Simple heuristic: assume small positive expected value per round.
        """
        if remaining_rounds <= 0:
            return 0.0
        
        # Conservative estimate: assume break-even or small loss per round
        # This encourages risk-averse play
        return remaining_rounds * (-5.0)  # Small expected loss per round
    
    def _copy_state(self, state: GameState) -> GameState:
        """Create a deep copy of game state"""
        return GameState(
            hole_cards_agent=state.hole_cards_agent.copy(),
            hole_cards_opponent=state.hole_cards_opponent.copy(),
            community_cards=state.community_cards.copy(),
            pot=state.pot,
            agent_stack=state.agent_stack,
            opponent_stack=state.opponent_stack,
            agent_bet_this_round=state.agent_bet_this_round,
            opponent_bet_this_round=state.opponent_bet_this_round,
            betting_history=state.betting_history.copy(),
            round_number=state.round_number,
            current_bet=state.current_bet,
            dealer_position=state.dealer_position,
            street=state.street,
            agent_acted=state.agent_acted,
            opponent_acted=state.opponent_acted,
            hand_over=state.hand_over
        )

