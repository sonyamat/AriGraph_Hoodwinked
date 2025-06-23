import os
import sys
import pandas as pd
import time
from environment import Game
import warnings
import argparse
# from numba import cuda
# import torch

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This script allows you to run a batch of games. Specify the games you'd like to run
in a file such as jobs/1.csv, and then run `python eval.py --job_number 1`. The results
will be saved in /results in a .csv file with a integer name. Temporary copies will also 
be saved, in order to prevent the loss of data if your machine stops running. 
"""

def run_job(
    num_games: int, 
    num_players: int, 
    impostor_agent: str, 
    innocent_agent: str, 
    discussion: bool, 
    start_location: str,
    eval_cols,
    impostor_type: str,
    innocent_type: str,
    job_name,
    device=0
):
    """
    Runs a number of games with the given specifications. 
    Returns a dictionary of eval results with a row for each player. 
    """
    # Initialize dictionary to be returned
    eval_dict = {colname: [] for colname in eval_cols}
    gpt = None

    # Run each game and store results
    for i in range(1, num_games+1):
        print("-"*15 + f"GAME {i}/{num_games}" + "-"*15)
        try:
            # Time the game
            start_time = time.time()
    
            # Define the game
            discussion = discussion
            game = Game(discussion=discussion, device=device, gpt=gpt)
    
            # Load the players into the game
            game.load_random_players(num_players, impostor_agent, innocent_agent, impostor_type, innocent_type)
    
            # Play the game
            player_dicts = game.play()
    
            # Record the runtime
            end_time = time.time()
            gpt = game.gpt
            runtime = end_time - start_time
    
            # Condense player dicts into a single dictionary
            for player_dict in player_dicts:
                for k in list(eval_dict.keys())[6:]:
                    if k in player_dict.keys():
                        eval_dict[k].append(player_dict[k])
                    else:
                        eval_dict[k].append("")
            
            # Store game-level information in eval_dict
            # Duplicated in each agent's row for ease of display
            # Memory inefficient -- change this if low on memory
            eval_dict["game_num"].extend([i for _ in range(num_players)])
            eval_dict["runtime"].extend([runtime for _ in range(num_players)])
            eval_dict["num_players"].extend([num_players for _ in range(num_players)])
            eval_dict["discussion"].extend([discussion for _ in range(num_players)])
            eval_dict["impostor_type"].extend([impostor_type for _ in range(num_players)])
            eval_dict["innocent_type"].extend([innocent_type for _ in range(num_players)])
    
            # Count API hits for monitoring
            # api_hits = sum(eval_dict['num_turns'][-num_players:]) + \
            #     sum([i if (type(i)!=str) else 0 for i in eval_dict['num_killed'][-num_players:]]) * 2 * num_players
            # print(f'api_hits: {api_hits}')
    
            # used_device = cuda.get_current_device()
            
            # used_device.reset()
    
            if i % 10 == 0:
                temp_save(eval_dict, job_name)

        # Catch errors and continue with the next game
        except Exception as e:
            print("Error: ", e.args)
            temp_save(eval_dict, job_name)
            time.sleep(20)
            continue

    return eval_dict


def temp_save(eval_dict, job_name):
    temp_save = pd.DataFrame(eval_dict)
    temp_save_path = get_save_path(job_name).replace(".csv", "_temp.csv")
    temp_save.to_csv(temp_save_path)


def get_save_path(job_name):
    """
    Returns a pathname to be used throughout the evaluation. 
    """
    return f"results/{job_name}.csv"
    # save_dir = 'results'
    # if not os.path.exists(save_dir): os.makedirs(save_dir)
    # file_name = str(len([name for name in os.listdir(save_dir)
    #                 if os.path.isfile(os.path.join(save_dir, name))]))
    # full_path = save_dir + '/' + file_name + '.csv'
    # return full_path

if __name__ == "__main__":
    # Read the command line argument for the job number
    # parser = argparse.ArgumentParser(description='Process the job number.')
    # parser.add_argument('--job_number', type=int, required=True, help='Which .csv file in the /jobs folder to run')
    # parser.add_argument('job_number', type=int)

    # args = parser.parse_args()
    # job_number = args.job_number

    job_name = "graph_players_exp_11"
    device = 0

    # Read the schedule of jobs
    schedule = pd.read_csv(f"jobs/{job_name}.csv")
    save_path = get_save_path(job_name)
    print(save_path)
    # save_path = f"{job_name}_1"
    
    # Set up the evaluation structure
    results_cols = [
        "game_num", "runtime", "num_players", "discussion", 'impostor_type', "innocent_type",
        "name", "agent", "killer", "num_turns", "banished",
        "killed", "escaped", "num_killed", "num_escaped", 
        "duplicate_search_rate", "vote_rate_for_self", "vote_rate_for_killer", 
        "witness_vote_rate_for_killer", "non_witness_vote_rate_for_killer",
        "story", "actions", "votes", "witness_during_vote", 'graph', 'graph_new_triplets_cnt', 'graph_replacements_cnt'    ]
    results = {colname: [] for colname in results_cols}

    # Run each job individually
    for idx, row in schedule.iterrows():
        eval_dict = run_job(
            num_games = row['num_games'],
            num_players = row['num_players'],
            impostor_agent = row['impostor_agent'],
            innocent_agent = row['innocent_agent'],
            impostor_type = row['impostor_type'],
            innocent_type = row['innocent_type'],
            discussion = row['discussion'],
            start_location = row['start_location'],
            eval_cols = results_cols,
            job_name=job_name,
            device = device
        )

        # Join all eval dicts into a single results dict
        for k, v in eval_dict.items():
            if k not in results.keys():
                results[k] = v
            else:
                results[k].extend(v)
        
        # Save results as .csv in results folder after each job
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)