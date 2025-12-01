"""
Basic test script to verify the implementation works correctly.
"""

from poker_game import PokerGame, HandEvaluator, Card
from pomdp_agent import POMDPAgent, BeliefState, OpponentModel
from evaluation import Evaluator

def test_hand_evaluation():
    """Test hand evaluation"""
    print("Testing hand evaluation...")
    evaluator = HandEvaluator()
    
    # Test pair
    hole = [Card(14, 0), Card(14, 1)]  # Pair of Aces
    community = [Card(10, 0), Card(9, 1), Card(8, 2)]
    hand = evaluator.evaluate_hand(hole, community)
    print(f"Pair of Aces: rank={hand[0]}, tie={hand[1]}")
    assert hand[0] == 1, "Should be a pair"
    
    # Test high card
    hole = [Card(14, 0), Card(13, 1)]  # Ace-King
    community = [Card(10, 0), Card(9, 1), Card(8, 2)]
    hand = evaluator.evaluate_hand(hole, community)
    print(f"Ace-King high: rank={hand[0]}, tie={hand[1]}")
    assert hand[0] == 0, "Should be high card"
    
    print("✓ Hand evaluation tests passed\n")

def test_game_basic():
    """Test basic game functionality"""
    print("Testing basic game functionality...")
    game = PokerGame(initial_stack=1000, small_blind=10, big_blind=20)
    
    # Start a hand
    state = game.start_new_hand(1000, 1000, 1, dealer=0)
    print(f"Initial pot: {state.pot}")
    print(f"Agent stack: {state.agent_stack}, Opponent stack: {state.opponent_stack}")
    assert state.pot == 30, "Pot should be 30 (10+20)"
    assert state.agent_stack == 990, "Agent should have 990 after small blind"
    assert state.opponent_stack == 980, "Opponent should have 980 after big blind"
    
    # Get legal actions
    actions = game.get_legal_actions(state, is_agent=True)
    print(f"Legal actions for agent: {len(actions)}")
    assert len(actions) > 0, "Should have legal actions"
    
    print("✓ Basic game tests passed\n")

def test_belief_state():
    """Test belief state initialization"""
    print("Testing belief state...")
    from poker_game import Deck
    
    deck = Deck()
    agent_hole = deck.deal(2)
    remaining = deck.cards
    
    belief = BeliefState(remaining, agent_hole)
    print(f"Belief initialized with {len(belief.opponent_hand_probs)} possible hands")
    assert len(belief.opponent_hand_probs) > 0, "Should have possible hands"
    
    # Sample a hand
    sampled = belief.sample_opponent_hand()
    print(f"Sampled opponent hand: {sampled}")
    assert len(sampled) == 2, "Should sample 2 cards"
    
    print("✓ Belief state tests passed\n")

def test_single_hand():
    """Test playing a single hand"""
    print("Testing single hand play...")
    game = PokerGame(initial_stack=1000, small_blind=10, big_blind=20)
    agent = POMDPAgent(num_rollouts=10)  # Reduced for speed
    opponent = OpponentModel(behavior_type='tight')
    
    state = game.start_new_hand(1000, 1000, 1, dealer=0)
    
    from poker_game import Deck
    deck = Deck()
    agent_hole = state.hole_cards_agent
    remaining = [c for c in deck.cards if not any(
        c.rank == h.rank and c.suit == h.suit for h in agent_hole
    )]
    belief = BeliefState(remaining, agent_hole)
    
    # Play a few actions
    max_iter = 10
    iter_count = 0
    
    while not state.hand_over and iter_count < max_iter:
        iter_count += 1
        
        if not state.agent_acted:
            action, amount = agent.choose_action(state, belief, remaining_rounds=19)
            state = game.apply_action(state, action, amount, is_agent=True)
        
        if not state.opponent_acted and not state.hand_over:
            opp_action, opp_amount = opponent.choose_action(game, state, state.hole_cards_opponent)
            state = game.apply_action(state, opp_action, opp_amount, is_agent=False)
            belief.update_from_action(opp_action, opp_amount, state.community_cards,
                                    state.pot, state.current_bet, state.betting_history)
    
    print(f"Hand completed after {iter_count} iterations")
    print(f"Final pot: {state.pot}")
    print("✓ Single hand test passed\n")

def test_evaluator():
    """Test evaluation metrics"""
    print("Testing evaluator...")
    evaluator = Evaluator()
    
    # Add some test results
    evaluator.add_result(100.0, [5.0] * 20)
    evaluator.add_result(-50.0, [-2.5] * 20)
    evaluator.add_result(200.0, [10.0] * 20)
    
    summary = evaluator.get_summary()
    print(f"Expected loss: {summary['expected_loss']:.2f}")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    assert summary['num_matches'] == 3, "Should have 3 matches"
    
    print("✓ Evaluator tests passed\n")

if __name__ == "__main__":
    print("="*60)
    print("Running Basic Tests")
    print("="*60 + "\n")
    
    try:
        test_hand_evaluation()
        test_game_basic()
        test_belief_state()
        test_single_hand()
        test_evaluator()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

