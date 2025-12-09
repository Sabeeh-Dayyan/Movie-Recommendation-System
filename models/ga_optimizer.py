#models/ga_optimizer.py
import pandas as pd
import random
import numpy as np

class GAOptimizer:
    def __init__(self, df, pop_size=20, generations=15, chrom_length=5, mutation_rate=0.2):
        # Make a copy to avoid modifying the original dataframe
        self.original_df = df.copy()
        
        # Only drop rows with NaN in essential columns
        essential_columns = ['title']  # Add only truly essential columns here
        self.df = df.copy().dropna(subset=essential_columns).reset_index(drop=True)
        
        if len(self.df) == 0:
            print("Warning: All rows were dropped due to NaN values in essential columns")
            # Use original df as fallback
            self.df = df.copy().reset_index(drop=True)
            # Fill NaN values with defaults
            if 'popularity' in self.df.columns:
                self.df['popularity'] = self.df['popularity'].fillna(self.df['popularity'].mean() if not self.df['popularity'].empty else 0)
            if 'vote_average' in self.df.columns:
                self.df['vote_average'] = self.df['vote_average'].fillna(self.df['vote_average'].mean() if not self.df['vote_average'].empty else 0)
            if 'runtime' in self.df.columns:
                self.df['runtime'] = self.df['runtime'].fillna(self.df['runtime'].mean() if not self.df['runtime'].empty else 0)
        
        # Print debug info
        print(f"Original dataframe shape: {df.shape}")
        print(f"Processed dataframe shape: {self.df.shape}")
        
        # Normalize features safely
        if 'popularity' in self.df.columns:
            max_pop = self.df['popularity'].max()
            if max_pop > 0:
                self.df['popularity'] = self.df['popularity'] / max_pop
            else:
                self.df['popularity'] = 0
                
        if 'vote_average' in self.df.columns:
            max_vote = 10  # Assuming vote_average is on a 0-10 scale
            self.df['vote_average'] = self.df['vote_average'] / max_vote
        
        if 'runtime' in self.df.columns:
            max_runtime = self.df['runtime'].max()
            if max_runtime > 0:
                self.df['runtime'] = self.df['runtime'] / max_runtime
            else:
                self.df['runtime'] = 0

        # Ensure we have enough data for the chromosome length
        self.POP_SIZE = pop_size
        self.NUM_GENERATIONS = generations
        self.CHROMOSOME_LENGTH = min(chrom_length, len(self.df))
        self.MUTATION_RATE = mutation_rate
        
        print(f"Chromosome length set to: {self.CHROMOSOME_LENGTH}")

    def fitness(self, chromosome):
        if not chromosome or len(chromosome) == 0:
            return 0
            
        # Validate chromosome indices
        valid_indices = [idx for idx in chromosome if 0 <= idx < len(self.df)]
        if not valid_indices:
            return 0
            
        selected = self.df.iloc[valid_indices]
        
        # Handle empty dataframes
        if len(selected) == 0:
            return 0
            
        # Calculate fitness with error handling
        fitness_score = 0
        if 'popularity' in selected.columns:
            fitness_score += 0.4 * selected['popularity'].mean()
        if 'vote_average' in selected.columns:
            fitness_score += 0.4 * selected['vote_average'].mean()
        if 'runtime' in selected.columns:
            fitness_score += 0.2 * selected['runtime'].mean()
            
        return fitness_score

    def generate_chromosome(self):
        # Generate a random chromosome with unique indices
        if len(self.df) == 0:
            return []
            
        if len(self.df) <= self.CHROMOSOME_LENGTH:
            # If we have fewer movies than chromosome length, use all available
            return list(range(len(self.df)))
        else:
            return random.sample(range(len(self.df)), self.CHROMOSOME_LENGTH)

    def crossover(self, parent1, parent2):
        if not parent1 or not parent2 or len(self.df) == 0:
            return self.generate_chromosome()
            
        cut = random.randint(1, len(parent1) - 1) if len(parent1) > 1 else 0
        child = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
        
        # Fill in any missing genes
        while len(child) < self.CHROMOSOME_LENGTH and len(self.df) > len(child):
            gene = random.randint(0, len(self.df) - 1)
            if gene not in child:
                child.append(gene)
                
        return child

    def mutate(self, chromosome):
        if not chromosome or len(self.df) == 0:
            return self.generate_chromosome()
            
        if random.random() < self.MUTATION_RATE and len(chromosome) > 0:
            i = random.randint(0, len(chromosome) - 1)
            
            # Try to find a gene not already in the chromosome
            attempts = 0
            while attempts < 10:  # Limit attempts to avoid infinite loop
                gene = random.randint(0, len(self.df) - 1)
                if gene not in chromosome:
                    chromosome[i] = gene
                    break
                attempts += 1
                
        return chromosome

    def run(self):
        # Ensure we have data to work with
        if len(self.df) == 0:
            raise ValueError("No movies available for optimization")
            
        # Print the first few rows to debug
        print("First few rows of dataframe:")
        print(self.df[['title']].head())
            
        # Adjust chromosome length if needed
        self.CHROMOSOME_LENGTH = min(self.CHROMOSOME_LENGTH, len(self.df))
        
        if self.CHROMOSOME_LENGTH == 0:
            raise ValueError("Chromosome length cannot be zero")
            
        # Generate initial population
        population = [self.generate_chromosome() for _ in range(self.POP_SIZE)]
        
        # Run GA for specified generations
        for _ in range(self.NUM_GENERATIONS):
            # Sort by fitness (descending)
            population = sorted(population, key=self.fitness, reverse=True)
            
            # Keep top performers
            next_gen = population[:2]
            
            # Generate new population
            while len(next_gen) < self.POP_SIZE:
                # Select parents from top performers
                if len(population) >= 10:
                    parent1, parent2 = random.sample(population[:10], 2)
                else:
                    parent1, parent2 = random.sample(population, min(2, len(population)))
                
                # Create and mutate child
                child = self.crossover(parent1, parent2)
                next_gen.append(self.mutate(child))
                
            population = next_gen
            
        # Get best chromosome
        best_chromosome = sorted(population, key=self.fitness, reverse=True)[0]
        
        # Return the movies from the best chromosome
        result = self.df.iloc[best_chromosome]
        print(f"Returning {len(result)} optimized movies")
        return result