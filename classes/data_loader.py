import numpy as np
import pandas as pd

class DataLoader:

    def load_bank_dataset(self, path):
        df = pd.read_csv(path, sep=';')

        df_pos = df[df['y'] == 'yes']
        df_neg = df[df['y'] == 'no']
        # Randomly sample the negative class to match the size of the positive class
        df_neg_sampled = df_neg.sample(n=len(df_pos), random_state=42)

        # Combine the positive class and the newly sampled negative class
        df = pd.concat([df_pos, df_neg_sampled])
        # Shuffle the combined dataset to mix the 'yes' and 'no' rows
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separate features (X) and target (y)
        X_raw = df.drop(columns=['y'])
        y_raw = df['y'].values

        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X_raw, drop_first=True).astype(float)

        # Map target 'y' from 'no'/'yes' to -1 / +1
        y = np.where(y_raw == 'yes', 1, -1)

        # Split into Training and Testing Sets (80% train, 20% test)
        np.random.seed(42)
        indices = np.random.permutation(len(X_encoded))
        split_idx = int(0.8 * len(X_encoded))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        X_train, X_test = X_encoded.iloc[train_idx].values, X_encoded.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features (Mean=0, Variance=1)
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        train_std[train_std == 0] = 1e-8 

        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        # Add Bias Term (Intercept) - a column of 1s
        #X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        #X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        return X_train, y_train, X_test, y_test, X_raw, y_raw
    
    def generate_synthetic_dataset(self, n_samples=500, n_test=200, n_features=10, noise_ratio=0.15):
    
        X_train = np.random.randn(n_samples, n_features) # Gaussian distribution
        
        # makes only first three features valueable, others are noise
        true_weights = np.array([2.5, -2.0, 1.0, -2.0, 5.0] + [0.0] * (n_features - 5))
        z = np.dot(X_train, true_weights) # dot product
        
        # Logistic Regression
        p_y1 = 1 / (1 + np.exp(-z))
        y_clean = np.where(p_y1 > 0.5, 1, -1) # assign the class base on the probability
        
        # Inject Label Noise
        y_noisy = y_clean.copy()
        n_noisy = int(n_samples * noise_ratio)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        y_noisy[noisy_indices] = y_noisy[noisy_indices] * -1 # Flip the label
        
        X_test = np.random.randn(n_test, n_features)
        z_test = np.dot(X_test, true_weights)
        p_y1_test = 1 / (1 + np.exp(-z_test))
        y_test = np.where(p_y1_test > 0.5, 1, -1)

        return X_train, y_noisy, y_clean, noisy_indices, X_test, y_test