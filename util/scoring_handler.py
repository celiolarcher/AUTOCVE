from sklearn import metrics

def load_scoring(score_call):
    if isinstance(score_call,str):
        if score_call in metrics.SCORERS:
            score_call=metrics.get_scorer(score_call)
        else:
            raise Exception("Keyword isn't in metric.SCORES")
    elif not callable(score_call):
       raise Exception("Invalid value for scoring parameter")
    
    return score_call

