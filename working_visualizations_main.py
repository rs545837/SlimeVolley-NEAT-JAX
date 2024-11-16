import neat
import gym
import slimevolleygym
import numpy as np
import pygame
import pickle
import os
import imageio
import graphviz
import matplotlib.pyplot as plt
from datetime import datetime

# Set training parameters
POPULATION_SIZE = 128
MAX_ITER = 500000
TEST_INTERVAL = 5

# Initialize pygame to capture keyboard input
pygame.init()

# Create directories for visualizations
os.makedirs("training_gifs", exist_ok=True)
os.makedirs("network_visualizations", exist_ok=True)

def capture_frame(env):
    """Capture the current frame from the environment"""
    return env.render(mode='rgb_array')

def draw_net(config, genome, generation=None, view=False):
    """Draw the neural network with graphviz"""
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')

    # Input nodes
    dot.attr('node', shape='box', style='filled', color='lightgray')
    input_labels = ['agent_x', 'agent_y', 'ball_x', 'ball_y', 
                   'ball_vx', 'ball_vy', 'opponent_x', 'opponent_y']
    
    for i, label in enumerate(input_labels):
        dot.node(f'input{i}', label)

    # Output nodes
    dot.attr('node', shape='box', style='filled', color='lightblue')
    output_labels = ['move_left', 'move_right', 'jump']
    for i, label in enumerate(output_labels):
        dot.node(f'output{i}', label)

    # Hidden nodes
    dot.attr('node', shape='circle', style='filled', color='lightgreen')
    for node_id, node in genome.nodes.items():
        if node_id not in range(len(input_labels)) and node_id not in range(len(output_labels)):
            activation = node.activation
            dot.node(f'hidden{node_id}', f'{activation}\n{node_id}')

    # Connections
    for conn_gene in genome.connections.values():
        if conn_gene.enabled:
            input_node = conn_gene.key[0]
            output_node = conn_gene.key[1]
            weight = conn_gene.weight
            
            from_node = f'input{input_node}' if input_node < len(input_labels) else f'hidden{input_node}'
            to_node = f'output{output_node}' if output_node < len(output_labels) else f'hidden{output_node}'

            color = 'red' if weight < 0 else 'green'
            width = str(abs(weight))
            
            dot.edge(from_node, to_node, color=color, penwidth=width)

    # Save the visualization
    suffix = f"gen_{generation}" if generation is not None else "final"
    filename = f"network_visualizations/network_{suffix}"
    dot.render(filename, cleanup=True)
    
    return dot if view else None

def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0

        for episode in range(5):
            observation = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action_values = net.activate(observation)
                action = [
                    np.clip(action_values[0], 0, 1),
                    np.clip(action_values[1], 0, 1),
                    np.clip(action_values[2], 0, 1)
                ]

                observation, reward, done, info = env.step(action)
                
                # Calculate combined reward (same as original)
                ball_x, ball_y = observation[2], observation[3]
                agent_x, agent_y = observation[0], observation[1]
                
                combined_reward = (
                    reward * 2.0 +
                    0.5 * (1 - (agent_x - ball_x)) +
                    (10.0 * (1 - abs(agent_y - ball_y)) if ball_y > 0.5 else 0) +
                    (1000.0 if reward > 0 else 0.0) +
                    (-10.0 if reward < 0 else 0.0) +
                    (5.0 if action[2] > 0.5 and ball_y > 0.5 else 0.0) +
                    (1.0 if agent_x > 0.5 else 0.0)
                )

                episode_reward += combined_reward

            total_reward += episode_reward

        genome.fitness = total_reward / 5

    env.close()

def render_best_genome(winner, config, generation=None, num_episodes=1):
    """Render and save GIF of the best genome's gameplay"""
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Create a unique timestamp for this recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for episode in range(num_episodes):
        frames = []
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Capture frame for GIF
            frames.append(capture_frame(env))
            
            # Get action from neural network
            action_values = net.activate(observation)
            action = [np.clip(val, 0, 1) for val in action_values]
            
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            # Optional: Slow down rendering for better visualization
            pygame.time.wait(10)

        print(f"Episode {episode + 1} finished with reward: {total_reward}")
        
        # Save the GIF
        gen_suffix = f"gen_{generation}" if generation is not None else "final"
        gif_filename = f"training_gifs/gameplay_{gen_suffix}_ep{episode+1}_{timestamp}.gif"
        imageio.mimsave(gif_filename, frames, fps=30)

    env.close()

class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config, save_interval=10):
        self.config = config
        self.generation = 0
        self.save_interval = save_interval
        self.fitness_history = []

    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        self.fitness_history.append(best_genome.fitness)
        
        if self.generation % TEST_INTERVAL == 0:
            print(f"\nRendering the best genome at generation {self.generation}...")
            # Create network visualization
            draw_net(config, best_genome, self.generation)
            # Create gameplay GIF
            render_best_genome(best_genome, self.config, self.generation, num_episodes=1)
            # Plot fitness history
            self.plot_fitness_history()

        if self.generation % self.save_interval == 0:
            with open(f"best_genome_gen_{self.generation}.pkl", "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved best genome at generation {self.generation}")

    def plot_fitness_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history)
        plt.title('Best Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.savefig(f'network_visualizations/fitness_history_gen_{self.generation}.png')
        plt.close()

    def complete_extinction(self):
        print("Population extinct.")

    def species_stagnant(self, sid, species):
        print(f"Species {sid} is stagnant with {len(species.members)} members.")

def run_neat(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(VisualizeBestGenomeReporter(config, save_interval=10))

    winner = p.run(eval_genomes, MAX_ITER)

    # Save final genome
    with open("best_genome_final.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Final best genome saved to 'best_genome_final.pkl'.")

    # Create final visualizations
    print("Creating final visualizations...")
    draw_net(config, winner)
    render_best_genome(winner, config, num_episodes=3)

if __name__ == '__main__':
    config_path = 'neat-config-slimevolley.ini'
    run_neat(config_path)
