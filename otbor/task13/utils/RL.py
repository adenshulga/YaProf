import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from numba import jit

import pydot
from IPython.display import Image, display
from io import BytesIO

def visualize_mdp(mdps, index, image_width, image_height):
    mdp = mdps[index]
    
    # Create a new directed graph in pydot
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')  # 'LR' for horizontal layout

    # Add nodes and edges to the graph
    for state, actions in mdp.items():
        graph.add_node(pydot.Node(state))  # Add states as nodes
        for action, outcomes in actions.items():
            for outcome in outcomes:
                prob = outcome['probability']
                next_state = outcome['next_state']
                reward = outcome['reward']
                edge_label = f"action={action}, p={prob}, r={reward:.2f}"
                # Adding the edge to the graph
                graph.add_edge(pydot.Edge(state, next_state, label=edge_label))

    # Create the graph
    png_str = graph.create_png(prog='dot')

    # Display the image with specified width and height
    sio = BytesIO(png_str)
    display(Image(data=sio.getvalue(), format='png', width=image_width, height=image_height))

class Env:

    def __init__(self, mdp):
        self.mdp = mdp
        self.state = 0
        self.gamma = 0.9

    def get_state(self):
        return self.state

    def step(self, action):
        info = self.mdp[str(self.state)][str(action)]
        p = []
        ns = []
        r = {}
        for outcome in info:
            p.append(outcome['probability'])
            ns.append(outcome['next_state'])
            r[outcome['next_state']] = outcome['reward']

        p_sum = sum(p)
        if p_sum != 1:
            p = [x/p_sum for x in p]

        next_state = np.random.choice(ns, p=p)
        self.state = next_state
        return r[next_state]
    
    def reset(self):
        self.state = 0
    

class Agent:

    def __init__(self, q_table):
        self.q = q_table
        self.R = 0
        self.gamma = 0.9
        self.total_actions = 0
    
    def act(self, env):
        q = self.q[str(env.get_state())]
        a = max(q, key=q.get)
        r = env.step(a)
        self.total_actions += 1
        self.R += self.gamma**self.total_actions*r
        return r
    
    def reset(self, q_table):
        self.q = q_table
        self.R = 0
        self.gamma = 0.9
        self.total_actions = 0

# @jit(nopython=False)  # Just-In-Time compilation for numeric computations
def play(ag, env, num_of_steps, num_of_games, q_table):
    # Optimized play function
    R = np.zeros(num_of_games)
    for i in range(num_of_games):
        env.reset()
        ag.reset(q_table)
        for _ in range(num_of_steps):
            ag.act(env)
        R[i] = ag.R
    return np.mean(R)

class MetaAgent:

    def __init__(self, mdps, q_tables, num_steps=100, num_games=100) -> None:
        self.mdps = mdps
        self.q_tables = q_tables
        self.num_steps = num_steps
        self.num_games = num_games

    def search(self, i):
        '''i is an index of MDP'''
        env = Env(self.mdps[i])
        q_initial = self.q_tables[i]
        ag = Agent(q_initial)
        points = get_q_to_iterate(q_initial)
        num_steps = self.num_steps

        if is_determined(self.mdps[i]):
            # print('the game is determined')
            num_games = 1
        else:
            num_steps = 30
            num_games = 20

        result = []

        for state in q_initial.keys():
            for action in q_initial[state].keys():
                # Optimized number of points to iterate
                for value in np.linspace(min(points), max(points), 10):  # Reduce the number of points
                    q = q_initial.copy()  # Ensure a copy is used
                    q[state][action] = value
                    result.append([state, action, value, play(ag, env, num_steps, num_games, q)])


        optimal_index = np.argmin(np.array(result)[:, 3])

        return result[optimal_index]

    def solve_task(self):
        def task(i):
            optimal = self.search(i)
            state = optimal[0]
            action = optimal[1]
            value = optimal[2]
            return {'state': state, 'action': action, 'value': value}

        num_threads = 11  # Adjust based on your machine's capability
        results = []
        save_interval = 50  # Save every 50 iterations

        # Process the first 400 tasks without parallelization
        for i in tqdm(range(400)):
            results.append(task(i))
            if i % save_interval == 0 and i > 0:
                with open(f'submit_{i}.json', 'w') as f:
                    json.dump(results, f)

        # Process remaining tasks with parallelization
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {executor.submit(task, i): i for i in range(400, len(self.mdps))}
            
            for future in tqdm(as_completed(future_to_index), total=len(self.mdps) - 400):
                i = future_to_index[future]
                results.append(future.result())
                if i % save_interval == 0:
                    with open(f'submit_{i}.json', 'w') as f:
                        json.dump(results, f)

        # Save the final results to a JSON file
        print('Evaluation has finished')
        with open('submit_final.json', 'w') as f:
            json.dump(results, f)



def get_q_to_iterate(q_table):
    all_q_values = [value for actions in q_table.values() for value in actions.values()]
    all_q_values.sort()

    midpoints = [(all_q_values[i] + all_q_values[i+1]) / 2 for i in range(len(all_q_values) - 1)]

    points = [all_q_values[0]-1] + midpoints + [all_q_values[-1] + 0.5]
    return points

def is_determined(mdp):
        all_probs = np.array([i['probability'] for state in mdp.values() for state_info in state.values() for i in state_info])
        return np.all(all_probs == 1.0)

