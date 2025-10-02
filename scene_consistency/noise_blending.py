import random
import itertools


def register_extrapolable_noise_blending(
    pipe, t_1, t_2, use_extrapolable_noise_blending=False,
):
    extrapolable_noise_blending_kwargs = {
        'use_extrapolable_noise_blending': use_extrapolable_noise_blending,
        't_1': t_1,
        't_2': t_2,
    }
    pipe.extrapolable_noise_blending_kwargs = extrapolable_noise_blending_kwargs
    

def generate_combinations(input_list, n=2, partition=False, K=3):
    """
    Generate all possible combinations of n elements from the given list.

    Parameters:
        input_list (list): The input list to generate combinations from.
        n (int): Number of elements in each combination (default is 2).

    Returns:
        list: A list of combinations, where each combination is a list.
    """
    if partition:
        results = []
        for _ in range(K):
            random.shuffle(input_list)
            breakpoint()
            result = [[input_list[i], input_list[i + 1]] for i in range(0, len(input_list), 2)]
            
            for tmp in result:
                results.append(tmp)
    else:
        # Generate all possible combinations
        combinations = list(itertools.combinations(input_list, n))
        
        # Convert each combination tuple to a list
        results = [list(comb) for comb in combinations]
    
    return results