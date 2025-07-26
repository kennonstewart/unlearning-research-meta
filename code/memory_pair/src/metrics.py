def regret(prediction: float, target: float) -> float:
    """
    Calculates the squared error regret for a single prediction.
    Args:
        prediction (float): The predicted value.
        target (float): The true value.
    Returns:
        float: The regret (squared error).
    """
    return (prediction - target) ** 2
