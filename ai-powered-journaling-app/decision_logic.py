# decision_logic.py

### PHQ scorers

def score_phq9(items):
    assert len(items) == 9
    total = sum(items)
    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    elif total <= 19:
        severity = "moderately severe"
    else:
        severity = "severe"
    return total, severity


def score_phq2(items):
    assert len(items) == 2
    total = sum(items)
    flag_full = total >= 3
    return total, flag_full

### GAD scorers

def score_gad7(items):
    assert len(items) == 7
    total = sum(items)
    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    else:
        severity = "severe"
    return total, severity


def score_gad2(items):
    assert len(items) == 2
    total = sum(items)
    flag_full = total >= 3
    return total, flag_full

### model risk tier helper

def classify_risk_from_model(pred_label, probs, si_label="SI", mh_label="MH"):
    si_prob = probs.get(si_label, 0.0)
    mh_prob = probs.get(mh_label, 0.0)

    if pred_label == si_label or si_prob >= 0.5:
        return "high"
    elif pred_label == mh_label or mh_prob >= 0.5 or si_prob >= 0.3:
        return "elevated"
    else:
        return "low"

### main decision function used by the app

def decide_next_step(
    model_label,
    model_probs,
    phq9_items=None,
    gad7_items=None,
):
    risk_tier = classify_risk_from_model(model_label, model_probs)

    result = {
        "risk_tier": risk_tier,
        "show_crisis_card": False,
        "show_phq_gad_prompt": False,
        "recommendation": "general_positive_feedback",
        "phq_summary": None,
        "gad_summary": None,
        "suicidality_flag": False,
    }

    if risk_tier == "high":
        result["show_crisis_card"] = True
        result["recommendation"] = "crisis_resources"

    if risk_tier in ["elevated", "high"]:
        result["show_phq_gad_prompt"] = True

    if phq9_items is not None:
        phq_total, phq_severity = score_phq9(phq9_items)
        phq_item9 = phq9_items[8]
        suicidality_flag = phq_item9 > 0
        result["phq_summary"] = {
            "total": phq_total,
            "severity": phq_severity,
            "item9_positive": bool(suicidality_flag),
        }
        result["suicidality_flag"] = bool(suicidality_flag)

        if suicidality_flag:
            result["show_crisis_card"] = True
            result["recommendation"] = "crisis_resources"
        elif phq_total >= 15:
            result["recommendation"] = "seek_professional_help"
        elif phq_total >= 10:
            result["recommendation"] = "consider_professional_help"
        elif phq_total >= 5:
            result["recommendation"] = "monitor_and_self_care"

    if gad7_items is not None:
        gad_total, gad_severity = score_gad7(gad7_items)
        result["gad_summary"] = {"total": gad_total, "severity": gad_severity}
        if gad_total >= 15:
            result["recommendation"] = "seek_professional_help"
        elif gad_total >= 10 and result["recommendation"] not in [
            "seek_professional_help",
            "crisis_resources",
        ]:
            result["recommendation"] = "consider_professional_help"

    return result