def compute_severity(effects, interaction_type):

    if interaction_type == "synergistic":
        return "high"

    if len(effects) >= 3:
        return "high"

    if effects:
        return "medium"

    return "low"


def compute_harm_score(pred, confidence, effects, interaction_type):

    score = 50

    if pred == 1:
        score -= 40

    score -= int(confidence * 40)

    score -= len(effects) * 10

    if interaction_type == "synergistic":
        score -= 20

    if interaction_type == "antagonistic":
        score -= 15

    return max(-100, min(100, score))