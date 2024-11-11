from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class FusionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, k=0.5, t=0):
        """
        Initialize with default values of hyperparameters k and t.
        
        Parameters:
        - k (float): weight for x1.
        - t (float): threshold adjustment.
        """
        self.k = k
        self.t = t

    def fit(self, X, y):
        """
        Fit method that sets the classes_ attribute required by sklearn utilities.
        
        Parameters:
        - X: array-like, feature matrix
        - y: array-like, target labels
        """
        # Set self.classes_ to the unique classes in y
        self.classes_ = np.unique(y)
        
        # Fit doesn't actually learn parameters in this custom model
        return self

    def predict_proba(self, combined_prob):
        """
        Predicts binary labels (0 or 1) based on the equation:
        y = x1 * k + (1 - k) * x2 - t

        Parameters:
        - X (array-like): input features, where X[:, 0] is x1 and X[:, 1] is x2.

        Returns:
        - predictions (array-like): binary predictions.
        """
        Radiomics_prob = combined_prob[:, :2]
        dl_prob = combined_prob[:, 2:]
        # Compute the model output
        y_prob = []
        Radiomics_prob_max = np.max(Radiomics_prob, axis=1)
        for index,prob_max in enumerate(Radiomics_prob_max):
            if prob_max>=self.t:
                y_prob.append(Radiomics_prob[index])
            else:
                y_prob.append(dl_prob[index] * self.k + (1 - self.k) * Radiomics_prob[index])
            
        y_prob = np.array(y_prob)
        # Apply a threshold at 0.5 for binary classification
        return y_prob

    def predict(self, combined_prob:np.array):
        
        proba = self.predict_proba(combined_prob)
        
        return (proba[:, 1] > 0.5).astype(int)
        