import numpy as np
import matplotlib.pyplot as plt

# Parameters
max_users = 15  # Maximum number of users to consider
N_bits = 1000  # Number of bits per user
initial_SNR_dB = 15  # Initial Signal-to-Noise Ratio in dB

# Generate random user codes
def generate_user_codes(N_users, N_bits):
    return np.random.randint(0, 2, size=(N_users, N_bits))

# Generating random channel noise
def generate_noise(N_users, N_bits):
    return np.random.normal(0, 1, size=(N_users, N_bits))

# Define fitness function (bit error rate)
def calculate_ber(user_codes, received_signal):
    decoded_bits = (user_codes * received_signal) > 0
    errors = np.sum(np.abs(decoded_bits - user_codes))
    total_bits = user_codes.size
    ber = errors / total_bits
    return ber


class GA:
    def __init__(self, population_size, max_generations):
        self.population_size = population_size
        self.max_generations = max_generations
        self.population = []

    def initialize_population(self, N_users):
        for _ in range(self.population_size):
            individual = np.random.normal(0, 1, size=(N_users, N_bits))
            self.population.append(individual)

    def evaluate_population(self, user_codes, noise, SNR_dB):
        fitness_scores = []
        for individual in self.population:
            received_signal = np.sum(individual * user_codes, axis=0) + noise
            ber = calculate_ber(user_codes, received_signal)
            fitness_scores.append(1 / (1 + ber))  # Fitness is inversely proportional to BER
        return fitness_scores

    def evolve(self, N_users, user_codes, noise):
        self.initialize_population(N_users)
        bers = []

        for generation in range(self.max_generations):
            total_interference = np.sum(np.abs(noise))
            SNR_dB = initial_SNR_dB - total_interference  # Adjust SNR based on total interference

            fitness_scores = self.evaluate_population(user_codes, noise, SNR_dB)

            # Select parents based on fitness scores
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Sorting in descending order
            parents = [self.population[i] for i in sorted_indices[:2]]

            # Generate offspring using crossover and mutation
            offspring = [(parents[0] + parents[1]) / 2 + np.random.normal(0, 1, size=(N_users, N_bits)) for _ in range(self.population_size)]

            # Replace population with offspring
            self.population = offspring

            # Record best solution in each generation
            best_solution = self.population[np.argmax(fitness_scores)]
            received_signal = np.sum(best_solution * user_codes, axis=0) + noise
            ber = calculate_ber(user_codes, received_signal)
            bers.append(ber)

        return bers

# Run simulations for different number of users
users_range = range(1, max_users+1)
final_bers = []

for N_users in users_range:
    print(f"Simulating for {N_users} users...")
    user_codes = generate_user_codes(N_users, N_bits)
    noise = generate_noise(N_users, N_bits)
    
    ga = GA(population_size=10, max_generations=10)
    bers = ga.evolve(N_users, user_codes, noise)
    final_bers.append(bers[-1])  # Record BER after all iterations

# Plot the results
plt.figure(figsize=(10, 5))

# Number of Users vs. BER
plt.subplot(1, 2, 1)
plt.plot(users_range, final_bers, marker='o')
plt.title('Number of Users vs. Final Bit Error Rate (BER)')
plt.xlabel('Number of Users')
plt.ylabel('Final Bit Error Rate (BER)')
plt.grid(True)

# Number of Iterations vs. BER
plt.subplot(1, 2, 2)
for i, N_users in enumerate(users_range):
    plt.plot(range(1, 11,), bers, label=f'{N_users} Users')
plt.title('Number of Iterations vs. Bit Error Rate (BER)')
plt.xlabel('Number of Iterations')
plt.ylabel('Bit Error Rate (BER)')
#plt.legend(title='Number of Users')
plt.grid(True)

plt.tight_layout()
plt.show()
