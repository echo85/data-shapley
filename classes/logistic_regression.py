import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.loss_history = []

    def sigmoid(self, z):
        """Standard logistic sigmoid function"""
        # np.clip prevents overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # logistic loss
    def log2_loss(self, X, y):
        """Computes empirical risk using logarithmic loss base 2 """
        g_x = X.dot(self.weights)
        # log2(1 + e^(-y * g(x)))
        loss = np.log2(1 + np.exp(-y * g_x))
        return np.mean(loss)

    def train(self, X, y):
        """Train the logistic regression model using Gradient Descent"""
        N, d = X.shape
        self.weights = np.zeros(d) # Initialize weights to zero
        
        for epoch in range(self.epochs):
            g_x = X.dot(self.weights) # w^T * x_t
            z = y * g_x
              
            # Update weights with GD
            self.weights += self.lr * X.T.dot(y * self.sigmoid(-z)) / (N * np.log(2))
            
            # Optional: Track the loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.log2_loss(X, y)
                self.loss_history.append(loss)
                #print(f"epoch {epoch}: {loss}")          

    def predict_proba(self, X):
        """Returns the predicted probability P(Y=1|X=x)"""
        return self.sigmoid(X.dot(self.weights))

    def predict(self, X, threshold=0.5):
        """Predict class label +1 or -1 based on a 0.5 probability threshold"""
        return np.where(self.predict_proba(X) >= threshold, 1, -1)
    
    def performance_score(self, X_train_sub, y_train_sub, X_test, y_test):
        """Train Logistic Regression on a subset and return its negative log-loss."""
        # Train normal Logistic Regression
        self.train(X_train_sub, y_train_sub)
        return -self.log2_loss(X_test, y_test)

    def compute_v_empty(self, y_test):
        """Calculate baseline performance V(empty) using the prior distribution."""
        # Calculate the proportion of the positive class (assuming labels are +1 and -1)
        prior_1 = np.mean(y_test == 1)
        
        # Clip to avoid log(0) errors
        p = np.clip(prior_1, 1e-15, 1 - 1e-15)
        
        # Standard log-loss formula: -log2(p) for positive class, -log2(1-p) for negative class
        loss = np.where(y_test == 1, -np.log2(p), -np.log2(1 - p))
        
        return -np.mean(loss)

    def tmc_shapley(self, X_train, y_train, X_test, y_test, tolerance=0.01):
        """
        Approximates Data Shapley values using Truncated Monte Carlo (TMC-Shapley).
        """
        n = len(X_train)
        phi = np.zeros(n) # Initialize Shapley values
        phi_history = []  # track convergence
        # Calculate performance V(D) using the entire training set
        v_D = self.performance_score(X_train, y_train, X_test, y_test)
        
        # Calculate baseline performance V(empty)
        v_empty = self.compute_v_empty(y_test)
        
        print(f"Baseline Score (Empty): {v_empty:.4f}")
        print(f"Full Dataset Score (All Data): {v_D:.4f}")

        # TMC-Shapley Loop
        t = 0
        mean_change = 0
        while (t < 10) or (mean_change > 1e-4):
            t = t + 1
            # Generate a random permutation of the training data indices
            pi = np.random.permutation(n)
            v_prev = v_empty
            
            for j in range(n):
                idx = pi[j]
                # Because adding those remaining points doesn't meaningfully move the needle away from $V(D)$, the algorithm can safely approximate the marginal contribution of the following elements as zero and stop computing to save time.
                # Truncation check: if performance is very close to V(D) - full datataset score, stop training
                if abs(v_D - v_prev) < tolerance:
                    v_curr = v_prev
                else:
                    # Get the subset of data up to index j
                    subset_indices = pi[:j+1]
                    X_sub = X_train[subset_indices]
                    y_sub = y_train[subset_indices]
                    
                    # Train and evaluate
                    v_curr = self.performance_score(X_sub, y_sub, X_test, y_test)
                
                # Marginal contribution
                marginal_contribution = v_curr - v_prev
                
                # Update the running average for the Shapley value of data point `idx`
                phi[idx] = ((t - 1) / t) * phi[idx] + (1 / t) * marginal_contribution
                v_prev = v_curr

            phi_history.append(phi.copy())
            
            if t >= 10:
                recent = np.array(phi_history[-5:])
                mean_change = np.mean(np.std(recent, axis=0))
                if mean_change < 1e-4:
                    print(f"Converged at iteration {t} with mean φ={np.mean(phi):.6f}, "f"stability={mean_change:.6f}")
                
            if t % 10 == 0:
                print(f"Completed {t} iterations... Shapley value average: {np.mean(phi):.8f},"f"stability={mean_change:.6f}")
                
        return phi, np.array(phi_history)