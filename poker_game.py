"""
Texas Hold'em Poker Game Environment
Implements the core game logic including card dealing, betting rounds, and hand evaluation.
"""

import random
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from typing import Any


class Action(Enum):
    """Poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True)
class Card:
    """Represents a playing card"""
    rank: int  # 2-14 (2-10, J=11, Q=12, K=13, A=14)
    suit: int  # 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs
    
    def __str__(self):
        ranks = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        suits = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
        rank_str = ranks.get(self.rank, str(self.rank))
        return f"{rank_str}{suits[self.suit]}"


class Deck:
    """Standard 52-card deck"""
    
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, n: int) -> List[Card]:
        """Deal n cards from the deck"""
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt


class HandEvaluator:
    """Evaluates poker hand strength"""
    
    @staticmethod
    def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[int]]:
        """
        Evaluate the best 5-card hand from hole + community cards.
        Returns (hand_rank, tie_breaker) where lower hand_rank is better.
        Hand ranks: 0=High Card, 1=Pair, 2=Two Pair, 3=Three of a Kind,
                   4=Straight, 5=Flush, 6=Full House, 7=Four of a Kind,
                   8=Straight Flush, 9=Royal Flush
        """
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            # Not enough cards yet, return high card evaluation
            ranks = sorted([c.rank for c in all_cards], reverse=True)
            return (0, ranks)
        
        # Generate all 5-card combinations
        from itertools import combinations
        best_rank = 10
        best_tie = []
        
        for combo in combinations(all_cards, 5):
            rank, tie = HandEvaluator._evaluate_five_cards(list(combo))
            if rank < best_rank or (rank == best_rank and tie > best_tie):
                best_rank = rank
                best_tie = tie
        
        return (best_rank, best_tie)
    
    @staticmethod
    def _evaluate_five_cards(cards: List[Card]) -> Tuple[int, List[int]]:
        """Evaluate exactly 5 cards"""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1
        is_straight = HandEvaluator._is_straight(ranks)
        
        # Royal Flush
        if is_straight and is_flush and ranks[0] == 14 and ranks[4] == 10:
            return (9, [])
        
        # Straight Flush
        if is_straight and is_flush:
            return (8, [ranks[0]])
        
        # Four of a Kind
        if counts == [4, 1]:
            four_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (7, [four_rank, kicker])
        
        # Full House
        if counts == [3, 2]:
            three_rank = [r for r, c in rank_counts.items() if c == 3][0]
            two_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return (6, [three_rank, two_rank])
        
        # Flush
        if is_flush:
            return (5, ranks)
        
        # Straight
        if is_straight:
            return (4, [ranks[0]])
        
        # Three of a Kind
        if counts == [3, 1, 1]:
            three_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (3, [three_rank] + kickers)
        
        # Two Pair
        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (2, pairs + [kicker])
        
        # Pair
        if counts == [2, 1, 1, 1]:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (1, [pair_rank] + kickers)
        
        # High Card
        return (0, ranks)
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight (handles A-2-3-4-5 low straight)"""
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) < 5:
            return False
        
        # Check regular straight
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                return True
        
        # Check A-2-3-4-5 low straight
        if 14 in sorted_ranks:  # Ace present
            low_ranks = [r if r != 14 else 1 for r in sorted_ranks]
            low_ranks = sorted(set(low_ranks))
            for i in range(len(low_ranks) - 4):
                if low_ranks[i+4] - low_ranks[i] == 4:
                    return True
        
        return False
    
    @staticmethod
    def compare_hands(hand1: Tuple[int, List[int]], hand2: Tuple[int, List[int]]) -> int:
        """
        Compare two hands. Returns 1 if hand1 wins, -1 if hand2 wins, 0 if tie.
        """
        rank1, tie1 = hand1
        rank2, tie2 = hand2
        
        if rank1 < rank2:
            return 1
        if rank1 > rank2:
            return -1
        
        # Same rank, compare tie breakers
        for t1, t2 in zip(tie1, tie2):
            if t1 > t2:
                return 1
            if t1 < t2:
                return -1
        
        return 0


@dataclass
class GameState:
    """Represents the current state of a poker hand"""
    hole_cards_agent: List[Card]
    hole_cards_opponent: List[Card]
    community_cards: List[Card]
    pot: int
    agent_stack: int
    opponent_stack: int
    agent_bet_this_round: int
    opponent_bet_this_round: int
    betting_history: List[Tuple[str, str, int]]  # (player, action, amount)
    round_number: int
    current_bet: int  # Current bet to call
    dealer_position: int  # 0=agent is dealer, 1=opponent is dealer
    street: str  # 'preflop', 'flop', 'turn', 'river'
    agent_acted: bool
    opponent_acted: bool
    hand_over: bool
    deck: Optional[Deck] = None  # Deck for this hand (to maintain consistency)


class PokerGame:
    """Main poker game environment"""
    
    def __init__(self, initial_stack: int = 1000, small_blind: int = 10, big_blind: int = 20):
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = HandEvaluator()
    
    def start_new_hand(self, agent_stack: int, opponent_stack: int, round_num: int, dealer: int = 0) -> GameState:
        """Start a new hand"""
        deck = Deck()
        
        # Deal hole cards
        agent_hole = deck.deal(2)
        opponent_hole = deck.deal(2)
        
        # Post blinds
        if dealer == 0:  # Agent is dealer
            agent_bet = self.small_blind
            opponent_bet = self.big_blind
            agent_stack -= self.small_blind
            opponent_stack -= self.big_blind
        else:  # Opponent is dealer
            agent_bet = self.big_blind
            opponent_bet = self.small_blind
            agent_stack -= self.big_blind
            opponent_stack -= self.small_blind
        
        pot = self.small_blind + self.big_blind
        current_bet = self.big_blind
        
        return GameState(
            hole_cards_agent=agent_hole,
            hole_cards_opponent=opponent_hole,
            community_cards=[],
            pot=pot,
            agent_stack=agent_stack,
            opponent_stack=opponent_stack,
            agent_bet_this_round=agent_bet,
            opponent_bet_this_round=opponent_bet,
            betting_history=[('opponent' if dealer == 0 else 'agent', 'blind', self.big_blind),
                            ('agent' if dealer == 0 else 'opponent', 'blind', self.small_blind)],
            round_number=round_num,
            current_bet=current_bet,
            dealer_position=dealer,
            street='preflop',
            agent_acted=(dealer == 1),  # Big blind acts first preflop
            opponent_acted=(dealer == 0),
            hand_over=False,
            deck=deck  # Save deck for consistent dealing
        )
    
    def get_legal_actions(self, state: GameState, is_agent: bool) -> List[Tuple[Action, Optional[int]]]:
        """Get legal actions for current player"""
        actions = []
        stack = state.agent_stack if is_agent else state.opponent_stack
        bet_this_round = state.agent_bet_this_round if is_agent else state.opponent_bet_this_round
        
        # Can always fold
        actions.append((Action.FOLD, None))
        
        # Can check if no bet to call
        if state.current_bet == bet_this_round:
            actions.append((Action.CHECK, None))
        else:
            # Can call
            call_amount = state.current_bet - bet_this_round
            if call_amount <= stack:
                actions.append((Action.CALL, None))
        
        # Can bet/raise (discretized bet sizes: 1x, 2x, 3x pot)
        if state.current_bet == bet_this_round:  # No bet yet, can bet
            max_bet = min(stack, state.pot * 3)
            if max_bet > 0:
                for multiplier in [1, 2, 3]:
                    bet_size = min(state.pot * multiplier, stack)
                    if bet_size > 0:
                        actions.append((Action.BET, bet_size))
        else:  # Bet exists, can raise
            min_raise = state.current_bet * 2 - bet_this_round
            max_raise = stack
            if min_raise <= max_raise:
                for multiplier in [1, 2, 3]:
                    raise_size = min(state.current_bet * multiplier, stack)
                    if raise_size > bet_this_round:
                        actions.append((Action.RAISE, raise_size))
        
        return actions
    
    def apply_action(self, state: GameState, action: Action, amount: Optional[int], is_agent: bool) -> GameState:
        """Apply an action and return new state"""
        new_state = GameState(
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
        
        if action == Action.FOLD:
            new_state.hand_over = True
            new_state.betting_history.append(('agent' if is_agent else 'opponent', 'fold', 0))
            return new_state
        
        if action == Action.CHECK:
            new_state.betting_history.append(('agent' if is_agent else 'opponent', 'check', 0))
            if is_agent:
                new_state.agent_acted = True
            else:
                new_state.opponent_acted = True
            
            # If both acted and bets are equal, advance street
            if new_state.agent_acted and new_state.opponent_acted:
                if new_state.agent_bet_this_round == new_state.opponent_bet_this_round:
                    new_state = self._advance_street(new_state)
            
            return new_state
        
        if action == Action.CALL:
            call_amount = new_state.current_bet - (new_state.agent_bet_this_round if is_agent else new_state.opponent_bet_this_round)
            if is_agent:
                new_state.agent_stack -= call_amount
                new_state.agent_bet_this_round = new_state.current_bet
            else:
                new_state.opponent_stack -= call_amount
                new_state.opponent_bet_this_round = new_state.current_bet
            
            new_state.pot += call_amount
            new_state.betting_history.append(('agent' if is_agent else 'opponent', 'call', call_amount))
            
            if is_agent:
                new_state.agent_acted = True
            else:
                new_state.opponent_acted = True
            
            # If both acted and bets are equal, advance street
            if new_state.agent_acted and new_state.opponent_acted:
                if new_state.agent_bet_this_round == new_state.opponent_bet_this_round:
                    new_state = self._advance_street(new_state)
            
            return new_state
        
        if action == Action.BET or action == Action.RAISE:
            bet_amount = amount
            if is_agent:
                new_state.agent_stack -= bet_amount
                new_state.agent_bet_this_round += bet_amount
                new_state.current_bet = new_state.agent_bet_this_round
            else:
                new_state.opponent_stack -= bet_amount
                new_state.opponent_bet_this_round += bet_amount
                new_state.current_bet = new_state.opponent_bet_this_round
            
            new_state.pot += bet_amount
            action_name = 'bet' if action == Action.BET else 'raise'
            new_state.betting_history.append(('agent' if is_agent else 'opponent', action_name, bet_amount))
            
            if is_agent:
                new_state.agent_acted = True
                new_state.opponent_acted = False  # Opponent needs to respond
            else:
                new_state.opponent_acted = True
                new_state.agent_acted = False  # Agent needs to respond
            
            return new_state
        
        return new_state
    
    def _advance_street(self, state: GameState) -> GameState:
        """Advance to next betting street (flop, turn, river)"""
        if state.deck is None:
            # Fallback: create new deck if not available
            state.deck = Deck()
            all_dealt = state.hole_cards_agent + state.hole_cards_opponent
            state.deck.cards = [c for c in state.deck.cards if not any(
                c.rank == d.rank and c.suit == d.suit for d in all_dealt
            )]
        
        if state.street == 'preflop':
            # Burn one card, then deal flop
            if len(state.deck.cards) > 0:
                state.deck.deal(1)  # Burn card
            state.community_cards = state.deck.deal(3)
            state.street = 'flop'
        elif state.street == 'flop':
            # Burn one card, then deal turn
            if len(state.deck.cards) > 0:
                state.deck.deal(1)  # Burn card
            state.community_cards.append(state.deck.deal(1)[0])
            state.street = 'turn'
        elif state.street == 'turn':
            # Burn one card, then deal river
            if len(state.deck.cards) > 0:
                state.deck.deal(1)  # Burn card
            state.community_cards.append(state.deck.deal(1)[0])
            state.street = 'river'
        elif state.street == 'river':
            # Showdown
            state.hand_over = True
            return state
        
        # Reset betting for new street
        state.agent_bet_this_round = 0
        state.opponent_bet_this_round = 0
        state.current_bet = 0
        state.agent_acted = (state.dealer_position == 1)  # Dealer acts first post-flop
        state.opponent_acted = (state.dealer_position == 0)
        
        return state
    
    def resolve_hand(self, state: GameState) -> Tuple[int, int]:
        """
        Resolve hand at showdown. Returns (agent_chips_won, opponent_chips_won).
        If someone folded, they lose their contribution.
        """
        if state.hand_over and len(state.community_cards) < 5:
            # Someone folded
            if state.agent_bet_this_round > state.opponent_bet_this_round:
                # Opponent folded
                return (state.pot, 0)
            else:
                # Agent folded
                return (0, state.pot)
        
        # Showdown - compare hands
        agent_hand = self.evaluator.evaluate_hand(state.hole_cards_agent, state.community_cards)
        opponent_hand = self.evaluator.evaluate_hand(state.hole_cards_opponent, state.community_cards)
        
        result = self.evaluator.compare_hands(agent_hand, opponent_hand)
        
        if result > 0:  # Agent wins
            return (state.pot, 0)
        elif result < 0:  # Opponent wins
            return (0, state.pot)
        else:  # Tie
            return (state.pot // 2, state.pot // 2)

