import neat
import gym
import slimevolleygym
import numpy as np
from neat.reporting import StdOutReporter
from neat.population import CompleteExtinctionException
import pickle

# Evaluation function for genomes in the population
def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")
    
    for genome_id, genome in genomes:
        # Create the neural network for the current genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        
        # Play against the built-in AI multiple times to evaluate performance
        for episode in range(5):
            observation = env.reset()
            done = False
            while not done:
                # Use the neural network to choose an action based on the observation
                action_values = net.activate(observation)
                
                # Map neural network output to the three action spaces:
                # SlimeVolley has a continuous action space, so we use tanh-like behavior.
                move_left = action_values[0]  # Move left intensity
                move_right = action_values[1]  # Move right intensity
                jump = action_values[2]  # Jump intensity

                # Create discrete actions based on thresholds, with smooth transitions
                action = [int(move_left > 0.5), int(move_right > 0.5), int(jump > 0.5)]
                
                # Step the environment with the chosen action
                observation, reward, done, info = env.step(action)
                total_reward += reward

        # Set the fitness of the genome based on the total reward earned
        genome.fitness = total_reward / 5  # Average reward over episodes

    env.close()

# Function to visualize the best genome in action
def render_best_genome(winner, config, num_episodes=3):
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the environment to visualize it
            
            # Activate the neural network to choose an action
            action_values = net.activate(observation)
            move_left = action_values[0]
            move_right = action_values[1]
            jump = action_values[2]

            action = [int(move_left > 0.5), int(move_right > 0.5), int(jump > 0.5)]
            
            # Step the environment with the chosen action
            observation, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1} finished with reward: {total_reward}")
    
    env.close()

# Run the NEAT algorithm
def run_neat(config_file):
    # Load the NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population
    p = neat.Population(config)

    # Add a reporter to display progress
    p.add_reporter(StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT for up to 50 generations
    winner = None
    try:
        winner = p.run(eval_genomes, 50)
    except CompleteExtinctionException:
        print("Complete extinction occurred in the population.")

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome for future use
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner

if __name__ == '__main__':
    # Path to the NEAT configuration file
    config_path = 'neat-config-slimevolley.ini'

    # Run NEAT to evolve agents to play SlimeVolley
    winner = run_neat(config_path)

    # Render the gameplay of the best genome
    print("Rendering the best genome in action...")
    render_best_genome(winner, neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                           neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path))
