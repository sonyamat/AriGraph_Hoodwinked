import random
import re
import re
from tqdm import tqdm
from agents.agent_base import Player

class RelGraphPlayer(Player):
    def __init__(self, name, killer, agent, start_location="random"):
        """
        Initializes a player with the given name and identity. 
        """
        super().__init__(name, killer, agent, start_location="random")
        self.graph = None
        self.graph_copy = None # to save the graph with relationships with all people, not only active players
        self.eval["graph"] = self.graph

    def get_action_prompt(self, action_prompt):
        self.update_graph('')
        graph_prompt = self.get_graph_prompt()
        return self.story + graph_prompt + action_prompt

    def get_discussion_prompt(self, discussion_log):
        self.update_graph(discussion_log)
        graph_prompt = self.get_graph_prompt()
        return self.story + discussion_log + graph_prompt

    def get_graph_prompt(self):
        prompt = "Based on previous actions of other players, you analysed your relationship with each of them on a scale from 1 (your enemy) to 10 (your friend):\n"
        for player, score in self.graph.items():
            prompt += f'{player} -- {score}\n'
        return prompt

    def update_graph(self, discussion_log: str):
        if not self.graph_copy:
            self.graph_copy = {name: 5 for name in self.players_list if name != self.name}
            self.graph = {name: 5 for name in self.players_list if name != self.name}

        # print(f'players: {self.players_list}]ngraph: {self.graph}')
            
        current_graph_data = '\n'.join([f'{player}: {score}' for player, score in self.graph.items()])
        think_prompt = f"{self.story + discussion_log}\n Let's take a moment to think. Give each person a rating based on their suspicion: how often did he lie, suspect you, or just behave badly.\n"
        new_graph = {}
        # print(self.name, 'ACCESSING')
        for player, score in self.graph.items():
            # print(player)
            player_prompt = f"What score would you give now to player {player} on a scale from 1 (your enemy) to 10 (your friend)?\n"
            gpt_long_score = self.get_player_analysis(think_prompt + player_prompt).replace('\n', '')
            gpt_long_prompt = f"What score is given to {player}, based on this statement (from 1 to 10)? \n\"\"{gpt_long_score}\"\"\nAnswer with just a single number.\n"
            new_graph[player] = self.get_graph_score(gpt_long_prompt)
            self.graph_copy[player] = new_graph[player]
            # print(gpt_long_prompt + str(new_graph[player]))
        assert set(self.graph.keys()) == set(new_graph.keys())
        # print(f"{self.name}'s graph: {new_graph}")
        self.graph = new_graph

    def get_player_analysis(self, prompt):
        response = ':' # To stop the model from speaking for someone else
        while ':' in response: # clean up this shit, make something more beautiful
            response = self.gpt.generate(
                prompt = prompt,
                max_tokens = 50, 
                model = self.model,
                # To limit GPT to providing one player's dialogue ## doesnt work :(
                stop_tokens = ['"'], system_prompt=self.system_prompt
            )
        return response

    def get_graph_score(self, action_prompt):
        """
        """
        # Mark state as awaiting_response
        self.awaiting_response = True

        # Parse action prompt for valid actions
        action_int_list = [n for n in range(1, 11)]
        action_dict = {i: str(i) for i in action_int_list}
        valid_action = False

        # Get and validate action
        while valid_action == False:
            # Get action
            if self.agent == "random":
                action_int = self.get_random_action(action_int_list)
            elif self.agent == "cli":
                action_int = self.get_cli_action(
                    action_int_list, action_prompt)
            elif self.agent == "gpt":
                action_int = self.get_gpt_graph_score(action_prompt, action_dict=action_dict)
            # Validate action
            # try:
            assert type(action_int) == int, \
                "Selected action is not an integer"
            assert action_int in action_int_list, \
                "Selected action is not in action_int_list"
            valid_action = True
            # except:
            #     print("Invalid action. Please try again.")

        return action_int

    def get_gpt_graph_score(self, action_prompt, argmax=False, action_dict=None):
        if action_dict is None:
            action_dict = self.extract_list_items(action_prompt)
        option_probs = self.gpt.get_probs(action_prompt, action_dict, self.model, system_prompt=self.system_prompt)
        if argmax:
            selected_action = max(option_probs, key=option_probs.get)
        else:
            # Sample an action from the distribution
            rand_val = random.random()
            total = 0
            for action, prob in option_probs.items():
                total += prob
                if rand_val <= total:
                    selected_action = action
                    break
            else:  # This executes if the for loop doesn't break, i.e., if no action was selected.
                selected_action = random.choice(list(option_probs.keys()))

        # Return the most likely token among the valid voting options
        return int(selected_action)

    def finalize_eval(self, killer_name):
        """
        After the game is over, the game runs this command for each player
        to compute the final evaluation metrics stored in the player.eval dict. 
        """
        super().finalize_eval(killer_name)
        self.eval['graph'] = self.graph_copy