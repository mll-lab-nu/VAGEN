def system_prompt(natural=False, sab=False, **kwargs):
    natural_rule = "Natural blackjack (Ace + 10-value card) gives 1.5x reward." if natural else "Natural blackjack gives standard 1.0 reward."
    sab_rule = "\nUsing Sutton & Barto rules: Player wins automatically with natural if dealer doesn't have natural." if sab else ""
    
    return f"""You are a Blackjack player.

Blackjack Quick Guide:
Goal: Get a hand value closer to 21 than the dealer without going over (busting).

Card Values:
- Number cards (2-9): Face value
- Face cards (Jack, Queen, King): 10 points each
- Ace: 1 or 11 points (whichever is better for your hand)

Game Rules:
1. You start with 2 cards, dealer shows 1 card (has 1 hidden)
2. You can "Hit" to take another card or "Stand" to keep your current hand
3. If you go over 21, you bust and lose immediately
4. If you stand, dealer reveals hidden card and hits until reaching 17+
5. Closest to 21 wins, ties are draws
6. {natural_rule}{sab_rule}

Available actions: "Hit" (take another card), "Stand" (keep current hand)
Think strategically about the risk vs. reward of taking another card.
"""

def init_observation_template(observation="", **kwargs):
    return f"""[Initial Hand]:
{observation}

Decide whether to hit (take another card) or stand (keep current hand).
"""

def action_template(valid_action=None, observation="", **kwargs):
    valid_action_str = ", ".join(valid_action) if valid_action else "None"
    return f"""After your answer, the extracted valid action: {valid_action_str}
Current state:
{observation}

Decide your next move: Hit or Stand.
"""

# Simple FORMAT_CONFIGS for Blackjack - matching the navigation pattern
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your reasoning, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": "<think>I have 16 and dealer shows 10. This is risky but I need to get closer to 21.</think><answer>Hit</answer>"
    },
    
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": "<answer>Hit</answer>"
    },
    
    "grounding": {
        "description": "You should first give your observation of the current game state, then your reasoning, and finally your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": "<think><observation>My hand totals 16, dealer is showing a 10-value card, no usable ace</observation><reasoning>With 16 against dealer 10, I'm in a difficult position but hitting gives me a chance to improve</reasoning></think><answer>Hit</answer>"
    },
    
    "worldmodeling": {
        "description": "You should first give your reasoning, then predict what state you will be in after your action, and finally your answer.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": "<think><reasoning>I have 16 against dealer 10, this is a tough spot</reasoning><prediction>If I hit, I will most likely bust and my hand will exceed 21, ending the game with a loss</prediction></think><answer>Hit</answer>"
    },
    
    "grounding_worldmodeling": {
        "description": "You should first give your observation of the current game state, then your reasoning, then predict what state you will be in after your action, and finally your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": "<think><observation>My hand totals 16, dealer is showing a 10-value card, no usable ace</observation><reasoning>This is a difficult position against a strong dealer card</reasoning><prediction>If I hit, I will most likely bust and have a hand over 21, losing the game immediately</prediction></think><answer>Hit</answer>"
    }
}

def format_prompt_generator(format_type):
    """Generate a prompt function for the specified format type."""
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}"""
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

if __name__ == "__main__":
    # Test the prompts
    print("System prompt:")
    print(system_prompt())
    print("\n" + "="*50 + "\n")
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=1, action_sep=",", add_example=True))
        print("\n" + "="*30 + "\n")