import random
import re
import re
from tqdm import tqdm
import time

class Player():
    def __init__(self, name, killer, agent, start_location="random"):
        """
        Initializes a player with the given name and identity. 
        """
        self.name = name
        self.killer = killer
        self.alive = True
        self.banished = False
        self.has_key = False
        self.escaped = False
        self.story = ""
        self.system_prompt = ""
        self.actions = []
        self.votes = []
        self.witness = False 
        self.witness_during_vote = []
        self.awaiting_response = False
        self.players_list = None
        self.valid_actions = []
        self.observations = []
        self.observation = ''
        self.n_prev=5
        # self.graph_guidance = graph_guidance
        # # self.graph_guidance = True
        # self.graph = {}
        # self.graph_copy = {} # to save the graph with relationships with all people, not only active players

        # Set the initial location
        if start_location == "random":
            self.location = random.choice(
                ["Bedroom", "Bathroom", "Kitchen", "Hallway"]
            )
        elif start_location in ["Bedroom", "Bathroom", "Kitchen", "Hallway"]:
            self.location = start_location
        else:
            assert False, f"Start location {start_location} not implemented."

        # Set agent and potentially model
        if "gpt" in agent:
            self.agent = "gpt"
            self.model = agent[4:]
        elif agent == "llama":
            self.agent = "gpt"
            self.model = agent
        else:
            self.agent = agent
        assert self.agent in ["cli", "random", "gpt", "api"], \
            f"Player of type {agent} is not implemented."

        # Tracks evaluation metrics
        self.eval = {
            "name": self.name,
            "agent": agent,
            "killer": self.killer,
            "num_turns": 0,
            "banished": False,
            "story": self.story,
            "actions": self.actions,
            "votes": self.votes,
            # "graph": self.graph
        }

        if not self.killer:
            self.eval.update({
                "killed": False,
                "escaped": False,
            })
        else:
            self.eval.update({
                "num_killed": 0,
                "num_banished": 0,
                "num_escaped": 0,
            })

    def load_gpt(self, gpt):
        """
        Saves a reference to GPT provided by the Game class.
        """
        self.gpt = gpt

    def get_action(self, action_prompt):
        """
        Returns an integer representing a valid action based on the
        num_valid_actions argument passed into the function. 

        Part of me would prefer to read this from the player's story, 
        but maybe that's unnecessarily complicated. 
        """
        # Mark state as awaiting_response
        self.awaiting_response = True

        # Parse action prompt for valid actions
        action_int_list = [
            int(n) for n in re.findall("[0-9]", 
            action_prompt.split("Possible Actions:")[-1])
        ]
        valid_action = False
        start_time = time.time()

        # Get and validate action
        while valid_action == False:
            if time.time() - start_time > 180:
                raise TimeoutError("action selection lasted > 3 min")
            # Get action
            if self.agent == "random":
                action_int = self.get_random_action(action_int_list)
            elif self.agent == "cli":
                action_int = self.get_cli_action(
                    action_int_list, action_prompt)
            elif self.agent == "gpt":
                action_int = self.get_gpt_action(action_prompt)

            # Validate action
            try:
                assert type(action_int) == int, \
                    "Selected action is not an integer"
                assert action_int in action_int_list, \
                    "Selected action is not in action_int_list"
                valid_action = True
            except:
                print("Invalid action. Please try again.")

        action_text = self.decode_action(action_prompt, action_int)

        # Store generated action in Player object
        self.actions.append(action_text)
        self.eval['num_turns'] += 1
        self.awaiting_response = False
        
        self.start_new_observation()
        return action_text

    def start_new_observation(self):
        self.observations.append(self.observation)
        if len(self.observations) > self.n_prev:
            self.observations = self.observations[1:]
        self.observation = ''

    def get_cli_action(self, action_list, action_prompt):
        print(self.story)
        print(action_prompt)
        print(f"Please input one of the following valid inputs: {action_list}")
        return int(input())

    def get_random_action(self, action_list):
        return int(random.choice(action_list))
    
    def extract_list_items(self, string):
        pattern = r'(\d+)\.\s+(.*)'
        list_items = {}
        for match in re.finditer(pattern, string):
            num = int(match.group(1))
            content = match.group(2).strip()
            list_items[num] = content
        return list_items

    def get_action_prompt(self, action_prompt):
        return self.story + action_prompt

    def get_discussion_prompt(self, discussion_log):
        return self.story + discussion_log

    def get_gpt_action(self, action_prompt, argmax=False, action_dict=None):
        if action_dict is None:
            action_dict = self.extract_list_items(action_prompt)
        prompt = self.get_action_prompt(action_prompt)
        option_probs = self.gpt.get_probs(prompt, action_dict, self.model, system_prompt=self.system_prompt)
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
    
    def store_api_action(self, action_prompt, action_int):
        action_text =  self.decode_action(action_prompt, action_int)
        self.actions.append(action_text)
        self.eval['num_turns'] += 1
        self.awaiting_response = False

    def decode_action(self, action_prompt, action_int):
        """
        Given an action prompt and the integer number of an action,
        returns the text description of that action.
        """
        start_description_idx = action_prompt.find(str(action_int) + ". ") + 2
        end_description_idx = action_prompt[start_description_idx:].find('\n') + start_description_idx
        action_text = action_prompt[start_description_idx:end_description_idx].strip()

        return action_text

    def get_statement(self, discussion_log):
        if self.agent == "random":
            statement = self.get_idk_statement()
        elif self.agent == "cli":
            statement = self.get_cli_statement(discussion_log)
        elif self.agent == "gpt":
            statement = self.get_gpt_statement(discussion_log)
        self.start_new_observation()
        return statement + '"\n'

    def get_idk_statement(self):
        return "I don't know who the killer is."

    def get_cli_statement(self, discussion_log):
        print(self.story)
        print(discussion_log)
        return input()

    def get_gpt_statement(self, discussion_log):
        prompt = self.get_discussion_prompt(discussion_log)
        response = ':' # To stop the model from speaking for someone else
        start_time = time.time()
        while ':' in response: # clean up this shit, make something more beautiful
            if time.time() - start_time > 180:
                raise TimeoutError("action selection lasted > 3 min")
            response = self.gpt.generate(
                prompt = prompt + f'\nYour name is {self.name}, and you should only speak for yourself. Provide only your raw statement without any formatting. Remember to tell who will you vote for and why.\n',
                max_tokens = 50, 
                model = self.model,
                # To limit GPT to providing one player's dialogue ## doesnt work :(
                stop_tokens = ['"'], system_prompt=self.system_prompt
            )
        return response
    
    def get_vote(self, vote_prompt):
        if self.agent == "random":
            vote_int = self.get_random_vote(vote_prompt)
        elif self.agent == "cli":
            vote_int = self.get_cli_vote(vote_prompt)
        elif self.agent == "gpt":
            vote_int = self.get_gpt_action(vote_prompt)

        # Return the name of the person voted for
        vote = self.decode_vote(vote_prompt, vote_int)

        # Record for eval
        self.votes.append(vote)
        self.witness_during_vote.append(self.witness)

        self.start_new_observation()
        return vote

    def get_random_vote(self, vote_prompt):
        option_nums = re.findall("[0-9]", vote_prompt)
        return random.choice(option_nums)

    def get_cli_vote(self, vote_prompt):
        print(self.story)
        print(vote_prompt)
        return input()

    def decode_vote(self, vote_prompt, vote_int):
        # Build a dictionary mapping vote numbers to player names
        voting_options = dict()
        option_nums = re.findall("[0-9]", vote_prompt)
        
        # each option_num is stored as a string
        for num in option_nums:
            start_idx = vote_prompt.find(num)+3
            end_idx = vote_prompt[start_idx:].find('\n') + start_idx
            end_idx = len(vote_prompt) if end_idx < start_idx else end_idx
            voting_options[num] = vote_prompt[start_idx:end_idx]
        
        # Return the name that was voted for
        return voting_options[str(vote_int)]

    def update_story(self, new_data: str):
        self.story += new_data
        self.observation += new_data

    def finalize_eval(self, killer_name):
        """
        After the game is over, the game runs this command for each player
        to compute the final evaluation metrics stored in the player.eval dict. 
        """

        # Save story in evaluation metrics
        self.eval['story'] = self.story
        self.eval['actions'] = self.actions
        self.eval['votes'] = self.votes
        self.eval['witness_during_vote'] = self.witness_during_vote
        # self.eval['graph'] = self.graph_copy

        # Voting Metrics
        if len(self.eval['votes']) > 0:
            # Calculate vote rate for self
            self.eval['vote_rate_for_self'] = \
                sum([1 for i in self.eval['votes'] if i==self.name]) \
                    / len(self.eval['votes'])

            # Calculate vote rate for killer
            self.eval['vote_rate_for_killer'] = \
                sum([1 for i in self.eval['votes'] if i==killer_name]) \
                    / len(self.eval['votes'])


        # Tally votes for the killer conditioned on witnessing a murder
        killer_witness_votes, killer_not_witness_votes = 0, 0
        for w, v in zip(self.witness_during_vote, self.votes):
            if w==True and v==killer_name:
                killer_witness_votes += 1
            elif w==False and v==killer_name:
                killer_not_witness_votes += 1

        # Calculate vote rate when the player has witnessed a murder
        witness_votes = sum(self.witness_during_vote)        
        if witness_votes!=0:
            self.eval['witness_vote_rate_for_killer'] = \
                killer_witness_votes / witness_votes

        # Calculate vote rate when the player has not witnessed a murder
        non_witness_votes = len(self.votes) - witness_votes
        if non_witness_votes!=0:
            self.eval['non_witness_vote_rate_for_killer'] = \
                killer_not_witness_votes / non_witness_votes
        
        # Search Metrics
        search_actions = [a for a in self.actions if "Search" in a]
        if self.killer!=True and len(search_actions)!=0:
            search_locations = [a[11:] for a in search_actions]
            search_duplicates = [True if search_locations[:i].count(l)>0 \
                                         else False for i, l in enumerate(search_locations)]
            self.eval['duplicate_search_rate'] = sum(search_duplicates) \
                                                    / len(search_duplicates)