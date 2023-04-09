# Training log of Phase 1 (Deep Q players)

The Deep Q network was trained over 60000 games, divided into 3 round, 20000 games each.

    - Only the step of choosing a card from the hand was trained.
    - The AI is hard-coded by rule to avoid discarding for 3 coins if at all possible.
    - When there's a choice of building card or building wonder, card is randomly chosen with 70% probability
    - It doesn't matter if the model chose card expecting to play it, it might just end up being used to build a wonder stage

## Common paramaters over all rounds

    - Learning rate 0.0001
    - Gamma 1.0, with rewards only being given at the end of games as final points
    - Target model gets updated every 200 games

## Round 1

    - 2-5 training players + 1 random player, making 3-6 player games
    - For training players, 90% moves are random, 10% moves are greedy

### Round 1 testing
    - Tested by pitting in 3 vs 3 against random players
    - In 500 rounds of alternating seats (0-1-0-1-0-1), Round 1 weights score 10+ points higher than random players
    - In 500 rounds of grouped seats (0-0-0-1-1-1), Round 1 weights score only ~2 points higher.
    - This is probably because actions are 90% random making forward planning moves pointless
		ie. building resource production now doesn't guarantee a building using that resource will be selected in the future,
			so build choices that generate points are favored over resources
    - Science tend to be ignored, since without a focus, science doesn't score well.

## Round 2

    - 2-4 training player + 2 random players, making 4-6 player games, seats are random
    - For training players, 70% moves are random, 30% moves are greedy

### Round 2 testing
    - Tested by pitting in 3 vs 3 against random players
    - In 500 rounds of alternating seats (0-2-0-2-0-2), Round 2 weights score ~7 points higher than random players
    - In 500 rounds of grouped seats (0-0-0-2-2-2), Round 2 weights score 10+ points higher.
    - Apparently 30% determinism was sufficient to teach the model some forward thinking
	- But science still remains ignored

## Round 3

    - 2-4 training players + 1-2 frozen players, making 3-6 player games, seats are random
    - Frozen players have equal chance of being random, round 1, or round 2.
    - Non-random players have 50% chance to be focused player, 50% to be unfocused player.
    - Focused players make 90% greedy moves and 10% random moves
    - Unfocused players make 30% greedy moves and 70% random moves

### Round 3 testing
    - The model still doesn't seem to have figured out how to play science

#### Test 1
    - Player 0-1-2-3 are pitted against each other in completely random matchups
    - Number of players can be 3-6. Each player can be any of 0-1-2-3.
    - Average scores over 1000 games are 32.38802083, 39.83902878, 41.2962963, 45.33717835

#### Test 2
    - Player 0-1-2-3 are pitted against each other in 4 player games, with exactly 1 type of player each
    - Average scores over 1000 games are 30.755, 38.689, 42.146, 46.558

#### Test 3
    - Player 3 is pitted in 1 vs many games against any number of 0s for 300 rounds, then 1s for 300 rounds, then 2s
    - 3 vs 0s: 51.45 vs 34.36
	- 3 vs 1s: 45.94 vs 38.45
	- 3 vs 2s: 44.66 vs 39.53

## Potential improvements

    - Model for 'play' step (card-wonder-discard)
    - Somehow have to make the model figure out how to play science. Rewards probably need to be redesigned for that purpose.
    - Potentially rewards could be given at each step instead of all at once at the end, but the step function will need to be reworked.

    
    

