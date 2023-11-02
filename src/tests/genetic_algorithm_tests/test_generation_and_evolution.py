import pytest

from genetic_algorithm.genetic_algorithm_scripts.generation_and_evolution import *


class TestClass:

    def test_generate_genome(self):

        models = {"ridge": total_models["ridge"],
                    "lasso": total_models["lasso"],
                    "elastic": total_models["elastic"],
                    "linear_svr": total_models["linear_svr"]}

        # Test random permutations for fast mode
        for i in range(10000):
            tic = randint(0, len(models) - 1)
            model_name, genome, genome_dict = generate_genome(tic, models)

        models = {"decision_tree": total_models["decision_tree"],
                    "k_neighbours": total_models["k_neighbours"],
                    "random_forest": total_models["random_forest"],
                    "xgboost": total_models["xgboost"]}

        # Test random permutations for normal mode
        for i in range(10000):
            tic = randint(0, len(models) - 1)
            model_name, genome, genome_dict = generate_genome(tic, models)

    
    def test_generate_population(self):

        sizes = [10, 20, 30, 40, 50]

        for size in sizes:

            # Test sizing in fast mode works
            population = generate_population(mode="fast", population_size=size)
            for key, _ in population.items():
                assert size == len(population[key])

            # Test sizing in normal mode works
            population = generate_population(mode="normal", population_size=size)
            for key, _ in population.items():
                assert size == len(population[key])


    def test_mutate(self):
        pass


    def test_mutate_model(self):

        counter = 0
        
        models = {"ridge": total_models["ridge"],
                    "lasso": total_models["lasso"],
                    "elastic": total_models["elastic"],
                    "linear_svr": total_models["linear_svr"]}

        # Test random permutations for fast mode
        for i in range(10000):
            tic = randint(0, len(models) - 1)
            _, genome, genome_dict = generate_genome(tic, models)

            new_genome, new_loadout = mutate_model(genome, genome_dict)

            if new_genome == genome:
                counter += 1
            if new_loadout == genome_dict:
                counter += 1

        models = {"decision_tree": total_models["decision_tree"],
                    "k_neighbours": total_models["k_neighbours"],
                    "random_forest": total_models["random_forest"],
                    "xgboost": total_models["xgboost"]}

        # Test random permutations for normal mode
        for i in range(10000):
            tic = randint(0, len(models) - 1)
            model_name, genome, genome_dict = generate_genome(tic, models)

            new_genome, new_loadout = mutate_model(genome, genome_dict)

            if new_genome == genome:
                counter += 1
            if new_loadout == genome_dict:
                counter += 1

        print(f"{counter} number of identical permutations")


    def test_mutate_genome(self):

        populations = generate_population("fast", 20)

        for _, population in populations.items():
            for genome in population:
                index = randint(1, len(genome["genome"]) - 2)
                mutate_genome(genome["genome"], index, genome["loadout"])
    

if __name__=="__main__":
    tester = TestClass()
    # tester.test_generate_genome()
    # tester.test_generate_population()
    # tester.test_mutate_model()
    tester.test_mutate_genome()

