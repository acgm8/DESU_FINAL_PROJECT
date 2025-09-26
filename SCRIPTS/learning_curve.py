import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(pipe, X, y, cv, scoring="neg_mean_squared_error", train_sizes=np.linspace(0.2, 1.0, 10), title="Learning Curve"):
   
    train_sizes, train_scores, test_scores = learning_curve(pipe, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, shuffle=True, random_state=42)

    # Convert scores to error if negative
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, "o-", label="Training error")
    plt.plot(train_sizes, test_mean, "s-", label="Validation error")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()