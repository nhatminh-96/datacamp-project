from sklearn.preprocessing import OrdinalEncoder
import numpy as np



def convert_height(height_str):
    """
    Converting height from feet to centimeters
    """
    foot, inches = height_str.split('-')
    height_cm = 30.48 * float(foot) + 2.54* float(inches) 
    return height_cm

# Converting str valued columns to numerical columns
class CountOrdinalEncoder(OrdinalEncoder):
    """Encode categorical features as an integer array
    usint count information.
    """
    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        self.handle_unknown = 'use_encoded_value'
        self.unknown_value = np.nan
        super().fit(X)
        X_list, _, _ = self._check_X(X)
        # now we'll reorder by counts
        for k, cat in enumerate(self.categories_):
            counts = []
            for c in cat:
                counts.append(np.sum(X_list[k] == c))
            order = np.argsort(counts)
            self.categories_[k] = cat[order]
        return self





