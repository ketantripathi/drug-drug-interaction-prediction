def generate_explanation(d1, d2, interaction_type, effects):

    if interaction_type == "antagonistic":
        return f"{d1} and {d2} share similar drug classes, leading to pathway competition and increased risk of {', '.join(effects[:2])}."

    if interaction_type == "synergistic":
        return f"{d1} and {d2} enhance each other's effects, increasing risk of {', '.join(effects[:2])}."

    return f"{d1} and {d2} show limited interaction but may cause {', '.join(effects[:2])}."