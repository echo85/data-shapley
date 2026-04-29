import numpy as np
class Experiment:

    def fractional_remove(self, model, X_train, y_train, X_test, y_test, shapley_values, fractional_range = 25, sorted_desc = False):
        # Fractional remove worst Shapley values from dataset
        n_samples = len(X_train)
        sorted_indices = np.argsort(shapley_values)
        if sorted_desc == True:
            sorted_indices = np.argsort(shapley_values)[::-1]

        accuracy = []
        # Fractional Remove
        for i in range(fractional_range):
            num_to_remove = int((i*10)/1000 * n_samples) 
            indices_to_keep_shapley = sorted_indices[num_to_remove:]
            X_train_shapley = X_train[indices_to_keep_shapley]
            y_train_shapley = y_train[indices_to_keep_shapley]
            model.train(X_train_shapley, y_train_shapley)
            # Evaluate
            predictions = model.predict(X_test)
            tmp_accuracy = np.mean(predictions == y_test)
            accuracy.append(tmp_accuracy)
            if i % 2 == 0:
                print(f"Model Fractional Removed:{(i*10)/1000:.2f} Test Accuracy: {tmp_accuracy * 100:.2f}%")
        
        return accuracy
    
    def random_remove(self, model, X_train, y_train, X_test, y_test, fractional_range = 25):
        n_samples = len(X_train)
        accuracy = []
        # Fractional Remove
        for i in range(fractional_range):
            all_indices = np.arange(n_samples)
            num_to_remove =  int((i*10)/1000 * n_samples) 
            np.random.shuffle(all_indices)
            indices_to_keep_random = all_indices[num_to_remove:]
            X_train_random = X_train[indices_to_keep_random]
            y_train_random = y_train[indices_to_keep_random]
            model.train(X_train_random, y_train_random)
            
            predictions = model.predict(X_test)
            tmp_accuracy = np.mean(predictions == y_test)
            accuracy.append(tmp_accuracy)
            if i % 2 == 0:
                print(f"Model Random Removed:{(i*10)/1000:.2f} Test Accuracy: {tmp_accuracy * 100:.2f}%")
        
        return accuracy
    
    def flipping_robustness(self, y_train, corruption_rate=0.1):
        """
        Evaluates robustness by intentionally mislabeling data and checking if 
        TMC-Shapley assigns them the lowest values.
        """
        print(f"--- Starting Label-Flipping Robustness Evaluation ---")
        n_samples = len(y_train)
        num_corrupted = int(n_samples * corruption_rate)
        
        # Randomly select indices to corrupt (flip labels)
        np.random.seed(42)
        corrupted_indices = np.random.choice(n_samples, num_corrupted, replace=False)
        
        # Create a noisy copy of the training labels
        y_train_noisy = np.copy(y_train)
        y_train_noisy[corrupted_indices] = -y_train_noisy[corrupted_indices] # Flip +1 to -1 and vice versa

        return y_train_noisy, corrupted_indices
    
    def robustness(self, model, X_train, y_train, X_test, y_test):

        noise_levels = []
        inspection_curves = []

        for i in range(1,3):
            rate = i * 0.05
            noise_levels.append(rate)
            
            if rate > 0:
                y_train_tmp_noisy, corrupted_indices = self.flipping_robustness(y_train=y_train, corruption_rate=rate)
            else:
                y_train_tmp_noisy = y_train
                corrupted_indices = []
                
            print(f"\nStarting TMC-Shapley valuation with noisy labels corrupted percentage of {rate*100:.0f}%...")
            
            shapley_values, _ = model.tmc_shapley(
                    X_train, 
                    y_train_tmp_noisy, 
                    X_test, 
                    y_test, 
                    tolerance=0.004
                )
                
            mask_noisy = np.zeros(len(y_train), dtype=bool)
            if len(corrupted_indices) > 0:
                mask_noisy[corrupted_indices] = True
            
            predictions = model.predict(X_test)
            tmp_accuracy = np.mean(predictions == y_test)
            
            print(f"Model Test Accuracy: {tmp_accuracy * 100:.2f}%")
            
            if rate > 0:
                # Sort indices by Shapley value (lowest to highest)
                sorted_indices = np.argsort(shapley_values)
                random_indices = np.random.permutation(len(y_train))

                total_corrupted = len(corrupted_indices)
                total_data = len(y_train)
                
                fraction_fixed_shapley = []
                fraction_fixed_random = []
                fraction_inspected = []
                
                corrupted_found = 0
                corrupted_found_random = 0
                for step, idx in enumerate(sorted_indices):
                    if mask_noisy[idx]:
                        corrupted_found += 1
                    
                    random_idx = random_indices[step]
                    if mask_noisy[random_idx]:
                        corrupted_found_random += 1

                    fraction_inspected.append((step + 1) / total_data)
                    fraction_fixed_shapley.append(corrupted_found / total_corrupted)
                    fraction_fixed_random.append(corrupted_found_random / total_corrupted)
                    
                inspection_curves.append({
                    'noise_level': rate,
                    'fraction_inspected': fraction_inspected,
                    'fraction_fixed_shapley': fraction_fixed_shapley,
                    'fraction_fixed_random': fraction_fixed_random
                })
            else:
                inspection_curves.append(None) # No mislabeled data to fix at 0% noise

        return inspection_curves
    
    