def predict_mechanism(interaction_type):

    if interaction_type == "synergistic":
        return "combined pharmacological amplification"

    if interaction_type == "antagonistic":
        return "competitive pathway interference"

    if interaction_type == "additive":
        return "shared pharmacological effect"

    return "no major interaction mechanism"