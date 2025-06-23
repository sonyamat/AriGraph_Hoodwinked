import random
import re
import re
import ast
from tqdm import tqdm
import json
from time import time

from agents.agent_base import Player
from graphs.contriever_graph import ContrieverGraph

# from utils.envs_cfg import ENV_NAMES, FIRST_OBS, MAIN_GOALS
from utils.win_cond import win_cond_clean_place, win_cond_clean_take
# from utils.textworld_adapter import TextWorldWrapper, graph_from_facts

from prompts.system_prompts import default_system_prompt, system_plan_agent, \
    system_action_agent_sub_expl, if_exp_prompt

from utils.utils import Logger, observation_processing, find_unexplored_exits, \
    simulate_environment_actions, action_processing, action_deprocessing

class AriGraphPlayer(Player):
    def __init__(self, name, killer, agent, start_location="random", device=0):
        """
        Initializes a player with the given name and identity. 
        """
        super().__init__(name, killer, agent, start_location="random")
        
        if killer:
            self.main_goal = f"""Your name is {self.name}. You are the killer. There are 4 other players trapped in this house with you. They are looking for a key to escape the house. You must kill them all before they escape. 

        Be careful to avoid suspicion. If someone sees you commit murder in the same room as them, they will tell the other players, and the group will vote to banish you. If you say the killer isn't from your group, everyone will probably suspect you. Tonight, deception is your friend.\n\n"""
        else:
            self.main_goal = f"""Your name is {self.name}. You are trapped in a house with a ruthless killer and other innocent players. Search for the key and escape the house before you're murdered.

        Pay attention to any clues about the identity of the killer. When someone is killed, you will have the chance to speak with the other players and vote to banish one person. Banish the killer or escape the house and you win. When you see person murdered in the same room as you, you should suspect the person who did it. REMEMBER: The killer IS one of you. If someone says the killer isn't from your group, he is probably the killer himself. If you suspect someone, tell about it right away and don't make the killer kill you right afterwards.\n\n"""

        self.current_plan = f'''{{
                                "main_goal": {self.main_goal},
                                "plan_steps": [
                                    {{
                                    "sub_goal_1": "Start the game",
                                    "reason": "You should start the game"
                                    }},
                                ],
                                }}'''

        self.n_prev, self.topk_episodic = 5, 2
        self.max_steps, self.n_attempts = 150, 1
        self.need_exp = True
        self.top_episodic = ''
        self.device = device
        
        self.subgraph = []
        self.log = Logger("arigraph")
        print("Logger initialized")

        self.step = 0
        self.locations = ['Kitchen', 'Hallway', 'Bedroom', 'Bathroom']
        self.previous_location = ''
        self.explore = False
        self.history = ''

    def load_gpt(self, gpt):
        """
        Saves a reference to GPT provided by the Game class.
        """
        self.gpt = gpt
            

    def get_action_prompt(self, action_prompt):
        if self.observation != '':
            self.update_graph()
        graph_prompt = self.get_graph_prompt()
        return graph_prompt + action_prompt

    def get_discussion_prompt(self, discussion_log):
        if self.observation != '':
            self.update_graph()
        graph_prompt = self.get_graph_prompt()
        print(f'GRAPH PROMPT:\n {graph_prompt}\n END OF GRAPH PROMPT\n')
        return graph_prompt + '\nNow it is your turn to discuss who shall be banished.'

    # def get_graph_prompt(self):
    #     prompt = "Based on previous actions of other players, you analysed your relationship with each of them on a scale from 1 (your enemy) to 10 (your friend):\n"
    #     for player, score in self.graph.items():
    #         prompt += f'{player}: {score}\n'
    #     return prompt

    def update_graph(self):
        # observation = observation_processing(observation)
        self.step += 1
        if self.step == 1:
            print("Creating graph object")
            self.graph = ContrieverGraph(system_prompt="You are a helpful assistant", device=self.device, gpt=self.gpt)
            print("SUCCESS: Graph object created!")
        observation = "Game step #" + str(self.step) + "\n" + self.observation 
        self.log("Observation: " + self.observation + '\nEND OF OBSERVATION\n')  
        
        observed_items = self.item_processing_scores()
        items = {key.lower(): value for key, value in observed_items.items()}
        self.log("Crucial items: " + str(items))

        self.subgraph, self.top_episodic = self.graph.update(self.observation, \
                                                   self.observations, \
                                                   plan=self.current_plan, \
                                                   prev_subgraph=self.subgraph, \
                                                   locations=list(self.locations), \
                                                   curr_location=self.location, \
                                                   previous_location=self.previous_location, \
                                                   action=(self.actions[-1] if len(self.actions)>0 else 'start'), \
                                                   log=self.log, \
                                                   items1 = items, 
                                                   topk_episodic=self.topk_episodic)
        self.log("Length of subgraph: " + str(len(self.subgraph)))
        self.log("Associated triplets: " + str(self.subgraph))
        self.log("Episodic memory: " + str(self.top_episodic))

        # log('Explore: ' + str(self.explore))
        # #Exploration
        # all_unexpl_exits = get_unexpl_exits(self.locations, self.graph) if self.explore else ""
        # if self.explore:
        #     self.log(all_unexpl_exits)

        self.log("Valid actions: " + str(self.valid_actions))

        self.planning() # updates self.current_plan

    def get_graph_prompt(self):
        # \n5. Your {topk_episodic} most relevant episodic memories from the past for the current situation: {top_episodic}.
        prompt = f'''\n1. Main goal: {self.main_goal}
    \n2. History of {self.n_prev} last observations and actions: {"\n".join(self.observations)} 
    \n3. Your current observation: {self.observation}
    \n4. Information from the memory module that can be relevant to current situation:  {self.subgraph}
    \n5. Your {self.topk_episodic} most relevant episodic memories from the past for the current situation: {self.top_episodic}.
    \n6. Your current plan: {self.current_plan}'''
    
        # if self.explore:
        #     prompt += f'''\n7. Yet unexplored exits in the environment: {all_unexpl_exits}'''
               
        # prompt += f'''\n\nPossible actions in current situation: {self.valid_actions}'''  
        # # t = 0.2 if need_exp else 1
        # t = 0.2
        # action0 = self.gpt.generate(prompt, t = t, system_prompt=system_action_agent_sub_expl)
        # log("Action: " + action0)
        
        # try:
        #     action_json = json.loads(action0)
        #     action = action_json["action_to_take"]
        # except:
        #     log("!!!INCORRECT ACTION CHOICE!!!")
        #     action = "look"
    
        return prompt


    def planning(self, all_unexpl_exits=''):
# \n5. Your {topk_episodic} most relevant episodic memories from the past for the current situation: {top_episodic}.
        prompt = f'''\n1. Main goal: {self.main_goal}
\n2. History of {self.n_prev} last observations and actions: {"\n".join(self.observations)} 
\n3. Your current observation: {self.observation}
\n4. Information from the memory module that can be relevant to current situation: {self.subgraph}
\n5. Your {self.topk_episodic} most relevant episodic memories from the past for the current situation: {self.top_episodic}.
\n6. Your previous plan: {self.current_plan}'''

        if self.explore:
            prompt += f'''\n7. Yet unexplored exits in the environment: {all_unexpl_exits}'''

        self.current_plan = self.gpt.generate(prompt, t=0.2, system_prompt=system_plan_agent, model=self.model, max_tokens=50)
        self.log("Plan: " + self.current_plan)
            

    def item_processing_scores(self):
        prompt = "####\n" + \
             "You are a retriever part of the agent system that navigates the environment in a text-based game.\n" + \
             "You will be provided with agents' observation, what it carries and a plan that it follows.\n" + \
             "Your task is to extract entities from this data that can later be used to queue the agent's memory module to find relevant information that can help to solve the task. Assign a relevance score from 1 to 2 to every entity, that will reflect the importance of this entity and potential memories connected to this entity for the current plan and goals of the agent. Do not extract items like 'west', 'east', 'east exit', 'south exit'. Pay attention to the main goal of the plan. \n\n" + \
             "Current observation: {}\n".format(self.observation) + \
             "Current plan: {}\n\n".format(self.current_plan) + \
             "Answer in the following format:\n" + \
             '{"entity_1": score1, "entity_2": score2, ...}\n' + \
             "Do not write anything else\n"
        response = self.gpt.generate(prompt, model=self.model, max_tokens=50)
        entities_dict = ast.literal_eval(response)
        return entities_dict
        
        

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

    def finalize_eval(self, killer_name):
        """
        After the game is over, the game runs this command for each player
        to compute the final evaluation metrics stored in the player.eval dict. 
        """
        super().finalize_eval(killer_name)
        self.eval['graph_new_triplets_cnt'] = self.graph.new_triplets_cnt
        self.eval['graph_replacements_cnt'] = self.graph.replacements_cnt