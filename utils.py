def rule_engine(data):
    score = 0

    if data["amount"] > 300000:
        score += 40
    if data["is_international"] == 1:
        score += 20
    if data["device_new"] == 1:
        score += 25
    if data["hour"] < 5 or data["hour"] > 22:
        score += 15

    return min(score, 100)


def behavioral_score(history, data):
    if len(history) < 2:
        return 0

    last_amounts = [h["amount"] for h in history[-5:]]
    avg = sum(last_amounts) / len(last_amounts)

    if data["amount"] > avg * 3:
        return 30

    return 10


def combine_scores(ml, rules, behavior):
    final = (ml * 0.6) + (rules * 0.3) + (behavior * 0.1)
    return min(final, 100)


def decision_engine(score):
    if score > 70:
        return "BLOCK 🚫"
    elif score > 40:
        return "REVIEW ⚠️"
    else:
        return "APPROVE ✅"