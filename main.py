import neat
import gym
import slimevolleygym
import numpy as np
import pygame
import pickle
import os

# Set training parameters
POPULATION_SIZE = 128    # Population size
MAX_ITER = 500           # Max number of generations
TEST_INTERVAL = 5        # Interval for testing/rendering the best genome

# Initialize pygame to capture keyboard input
pygame.init()

# Evaluation function for genomes in the population
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
                # Optional: Normalize observations
                # observation = normalize_observation(observation)

                action_values = net.activate(observation)

                # Interpret network outputs as continuous actions
                move_left = np.clip(action_values[0], 0, 1)   # Ensure values are between 0 and 1
                move_right = np.clip(action_values[1], 0, 1)
                jump = np.clip(action_values[2], 0, 1)
                action = [move_left, move_right, jump]

                # Debugging: Print the chosen action
                # print(f"Genome {genome_id} - Action: {action}")

                # Send the action to the environment
                observation, reward, done, info = env.step(action)

                # Extract necessary information from observation
                # Example observation structure (verify with env):
                # [agent_x, agent_y, ball_x, ball_y, agent_velocity_x, agent_velocity_y,
                #  ball_velocity_x, ball_velocity_y, ...]
                if len(observation) < 12:
                    # Handle unexpected observation format
                    print(f"Unexpected observation format: {observation}")
                    continue

                ball_x = observation[2]
                agent_x = observation[0]

                # Reward shaping: Encourage proximity to the ball
                distance_to_ball = abs(agent_x - ball_x)
                proximity_reward = 1.0 / (distance_to_ball + 1)

                # Penalize unnecessary movements
                movement_penalty = 0.0
                if move_left > 0.5 and ball_x > agent_x:
                    movement_penalty -= 0.2  # Moving left when ball is to the right
                if move_right > 0.5 and ball_x < agent_x:
                    movement_penalty -= 0.2  # Moving right when ball is to the left

                # Combine environment reward with custom rewards
                combined_reward = reward + proximity_reward + movement_penalty
                episode_reward += combined_reward

            total_reward += episode_reward

        # Assign fitness as the average reward over episodes
        genome.fitness = total_reward / 5

    env.close()

def normalize_observation(observation):
    """
    Normalizes the observation data to a range suitable for neural network processing.
    Adjust the normalization based on actual observation ranges.
    """
    # Example normalization: Assuming all observation values range between -10 and 10
    # Modify the range based on actual data
    return np.clip(observation, -10, 10) / 10.0

# Function to visualize the best genome
def render_best_genome(winner, config, num_episodes=1):
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        frame = 0

        while not done:
            env.render()
            frame += 1

            # Interpret network outputs as continuous actions
            action_values = net.activate(observation)
            move_left = np.clip(action_values[0], 0, 1)   # Ensure values are between 0 and 1
            move_right = np.clip(action_values[1], 0, 1)
            jump = np.clip(action_values[2], 0, 1)
            action = [move_left, move_right, jump]

            # Debugging: Print the chosen action
            # print(f"Rendering Episode {episode + 1} - Action: {action}")

            # Send the action to the environment
            observation, reward, done, info = env.step(action)
            total_reward += reward

            # Optional: Slow down rendering for better visualization
            pygame.time.wait(10)

        print(f"Episode {episode + 1} finished with reward: {total_reward}")

    env.close()

# Custom Reporter to render the best genome after every TEST_INTERVAL generations
class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config, save_interval=50):
        self.config = config
        self.generation = 0
        self.save_interval = save_interval

    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        if self.generation % TEST_INTERVAL == 0:
            print(f"\nRendering the best genome at generation {self.generation}...")
            render_best_genome(best_genome, self.config)

        if self.generation % self.save_interval == 0:
            with open(f"best_genome_gen_{self.generation}.pkl", "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved best genome at generation {self.generation}")

    def complete_extinction(self):
        print("Population extinct.")

    def species_stagnant(self, species):
        print(f"Species {species.key} is stagnant.")

# Run NEAT evolution
def run_neat(config_file):
    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    # Load the NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population
    p = neat.Population(config)

    # Add reporters for logging
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add the custom visualization reporter with saving capability
    p.add_reporter(VisualizeBestGenomeReporter(config, save_interval=50))

    # Run NEAT for a defined number of generations
    winner = p.run(eval_genomes, MAX_ITER)

    # Save the best genome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved to 'best_genome.pkl'.")

    # Render the best genome's gameplay at the end
    print("Rendering the final best genome...")
    render_best_genome(winner, config, num_episodes=3)

if __name__ == '__main__':
    config_path = 'neat-config-slimevolley.ini'
    run_neat(config_path)
