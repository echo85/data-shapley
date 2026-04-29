import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Graph:
    def bank_projection(self, X_train,y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train_scaled)

        plt.figure(figsize=(10, 8))

        # Plot the majority class ('no', y=-1)
        plt.scatter(
            X_train_2d[y_train == -1, 0], 
            X_train_2d[y_train == -1, 1], 
            color='steelblue', 
            label='Did not subscribe (y=-1)', 
            alpha=0.4, 
            s=15
        )

        # Plot the minority class ('yes', y=1) on top so they are visible
        plt.scatter(
            X_train_2d[y_train == 1, 0], 
            X_train_2d[y_train == 1, 1], 
            color='darkorange', 
            label='Subscribed (y=1)', 
            alpha=0.8, 
            s=15
        )

        plt.title('2D PCA Projection of the Bank Marketing Training Data', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def synthetic_projection(self, X_train_synt, y_train_synt):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_synt)
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train_scaled)

        plt.figure(figsize=(10, 8))

        # Plot the majority class ('no', y=-1)
        plt.scatter(
            X_train_2d[y_train_synt == -1, 0], 
            X_train_2d[y_train_synt == -1, 1], 
            color='steelblue', 
            label='Did not subscribe (y=-1)', 
            alpha=0.4, 
            s=15
        )

        # Plot the minority class ('yes', y=1) on top so they are visible
        plt.scatter(
            X_train_2d[y_train_synt == 1, 0], 
            X_train_2d[y_train_synt == 1, 1], 
            color='darkorange', 
            label='Subscribed (y=1)', 
            alpha=0.8, 
            s=15
        )

        plt.title('2D PCA Projection of Synthetic Training Data', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def convergence(self, shapley_values, phi_history):

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)

        # Track how estimates stabilize for a sample of points
        sample_idx = np.random.choice(len(shapley_values), 10)
        for idx in sample_idx:
            plt.plot(phi_history[:, idx], alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("φ estimate")
        plt.title("Individual value convergence")

        plt.subplot(1, 2, 2)
        # Track standard deviation of changes across iterations
        stds = [np.std(phi_history[t] - phi_history[t-1]) 
                for t in range(1, len(phi_history))]
        plt.plot(stds)
        plt.xlabel("Iteration")
        plt.ylabel("Std of Δφ")
        plt.title("Convergence rate")
        plt.axhline(1e-4, color='r', linestyle='--', label='threshold')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def remove(self, accuracy_fractional, accuracy_random, metric= "Accuracy", title = 'Impact of Data Removal'):
        x_percent = [i for i in range(len(accuracy_fractional))]
        y1 = accuracy_fractional
        y2 = accuracy_random
        y1_percent = [val * 100 for val in y1]
        y2_percent = [val * 100 for val in y2]

        plt.figure(figsize=(8, 5))
        plt.plot(x_percent, y1_percent, marker='o', linestyle='-', color='b', label='Shapley Remove')
        plt.plot(x_percent, y2_percent, marker='o', linestyle='-', color='y', label='Random Remove')
        plt.title(title, fontsize=14)
        plt.xlabel('Fraction of Removed Data (%)', fontsize=12)
        plt.ylabel(f'{metric} (%)', fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    def shapley_distribution(self, shapley_values):
        plt.figure(figsize=(10, 2))

        # Create an array of zeros for the Y-axis to keep everything on one line
        y_zeros = np.zeros_like(shapley_values)

        plt.scatter(shapley_values, y_zeros, color='#377eb8', s=20, alpha=0.5)
        plt.axvline(x=0, color='dimgray', linestyle='--', linewidth=1.5)

        # Clean up the axes to make it look like a 1D graph
        ax = plt.gca()
        ax.get_yaxis().set_visible(False) # Hide the Y-axis completely
        
        # Hide all spines (borders) except the bottom one
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_edgecolor('gray')

        plt.title(f'TMC-Shapley Values Distribution ({len(shapley_values)} Points)', fontsize=14, pad=15)
        plt.xlabel('Calculated Shapley Value', fontsize=12)
        plt.tight_layout()
        plt.show()

    def noisy_data_removal(self, accuracy_noisy_fractional, accuracy_noisy_random):
        x_percent = [i for i in range(len(accuracy_noisy_fractional))]

        y1 = accuracy_noisy_fractional
        y2 = accuracy_noisy_random
        y1_percent = [val * 100 for val in y1]
        y2_percent = [val * 100 for val in y2]
        plt.figure(figsize=(8, 5))
        plt.plot(x_percent, y1_percent, marker='o', linestyle='-', color='b', label='Fractional Remove (20% Noise)')
        plt.plot(x_percent, y2_percent, marker='o', linestyle='-', color='y', label='Random Remove (20% Noise)')
        plt.title('Impact of Noisy Dataset on Model Accuracy', fontsize=14)
        plt.xlabel('Fraction of Removed Data (%)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)


        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    def inspection_curves(self, inspection_curves):
        plt.figure(figsize=(8, 6))
        
        # Track if we've added the random label to avoid duplicating it in the legend
        random_label_added = False
        
        for curve_data in inspection_curves:
            if curve_data is not None: 
                rate = curve_data['noise_level']
                
                x = np.array(curve_data['fraction_inspected']) * 100
                y_shapley = np.array(curve_data['fraction_fixed_shapley']) * 100
                y_random = np.array(curve_data['fraction_fixed_random']) * 100
                
                # Plot TMC-Shapley curve and capture its color
                line = plt.plot(x, y_shapley, linewidth=2, label=f'TMC-Shapley ({rate*100:.0f}% Noise)')
                curve_color = line[0].get_color()
                
                # Plot the corresponding empirical random curve using the same color but dotted
                random_label = 'Random Inspection (Empirical)' if not random_label_added else None
                plt.plot(x, y_random, linestyle=':', color=curve_color, alpha=0.7, label=random_label)
                
                random_label_added = True

        plt.title('Identifying Mislabeled Data: Data Shapley vs. Random Draw', fontsize=14, pad=15)
        plt.xlabel('Fraction of data inspected (%)', fontsize=12)
        plt.ylabel('Fraction of incorrect labels fixed (%)', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        plt.show()