"""Pure, score-independent run budget calculation."""
import math


def resolve_budget(cfg, measured):
    """Pure budgeting logic: no score-dependent or per-arm search."""
    b = cfg["budget"]
    factor = b["safety_factor"]
    seeds = len(cfg["ner"]["seeds"])
    per_epoch = sum(x["train"] + x["dev"] for x in measured["ner"].values()) * seeds * factor
    test_cost = sum(x["test"] for x in measured["ner"].values()) * seeds * factor
    maximum = cfg["ner"]["max_epochs"]
    affordable = math.floor((b["ner_hours"] * 3600 - test_cost) / max(per_epoch, 1e-6))
    epochs = min(maximum, affordable)
    minimum = min(10, maximum)
    reasons = []
    if epochs < minimum:
        reasons.append(f"NER budget cannot fit {minimum} epochs for every arm/seed")
    feature_cost = sum(measured["features"].values()) * factor
    if feature_cost > b["feature_hours"] * 3600:
        reasons.append("Feature extraction exceeds its allocated budget")
    mean_words = min(x["words_per_step"] for x in measured["lm"].values())
    maximum_steps = max(1, math.ceil(cfg["lm"]["max_tokens"] / mean_words))
    # Bound all scheduled validations, including the final one, conservatively.
    validation_count = math.ceil(cfg["lm"]["max_tokens"] / cfg["lm"]["eval_every_tokens"]) + 1
    validation_cost = sum(x["validation_seconds"] for x in measured["lm"].values()) * validation_count * factor
    training_seconds = sum(x["step_seconds"] for x in measured["lm"].values()) * factor
    steps = min(maximum_steps, math.floor((b["lm_hours"] * 3600 - validation_cost) / max(training_seconds, 1e-6)))
    if steps < 1:
        reasons.append("LM budget cannot fit validation and one paired optimizer step")
    estimated = (b["pilot_hours"] + b["reserve_hours"]) * 3600 + feature_cost + max(epochs, 0) * per_epoch + test_cost + max(steps, 0) * training_seconds + validation_cost
    if estimated > b["hours"] * 3600:
        reasons.append("Estimated total exceeds total budget")
    return {"ready": not reasons, "reasons": reasons, "lm_steps": max(0, steps), "ner_max_epochs": max(0, epochs),
            "estimated_seconds": estimated, "feature_seconds": feature_cost, "ner_seconds": max(epochs, 0) * per_epoch + test_cost,
            "lm_seconds": max(steps, 0) * training_seconds + validation_cost}

