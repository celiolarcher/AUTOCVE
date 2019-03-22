from sklearn.feature_selection import RFE

class RFE_Percentile(RFE):
    def fit(self, X, y, **kwargs):
        if(self.n_features_to_select is not None):
            coef_to_select=self.n_features_to_select
            self.n_features_to_select=int((coef_to_select/100)*X.shape[1])
            super(RFE_Percentile, self).fit(X,  y, **kwargs)
            self.n_features_to_select=coef_to_select
        else:
             super(RFE_Percentile, self).fit(X,  y, **kwargs)

        return self
