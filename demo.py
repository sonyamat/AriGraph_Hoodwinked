from agents.agent_base import Player
from agents.agent_relgraph import RelGraphPlayer
from agents.agent_arigraph import AriGraphPlayer
from environment import Game

# Define the game
device=0
game = Game(device=device)

# Load the players into the game
game.load_players([
    Player("Alice", killer=False, agent="llama"),
    Player("Bob", killer=True, agent="llama"),
    Player("Jim", killer=False, agent="llama"),
    Player("Adam", killer=False, agent="llama"),
    Player("Bill", killer=False, agent="llama")
    # Player("Bob", killer=False, agent="gpt-3.5"),
    # Player("Adam", killer=True, agent="cli"),
    # Player("Jim", killer=False, agent="gpt-3.5"),
    # Player("Alice", killer=False, agent="gpt-3.5"),
])

# Play the game
game.play()
