import data
import visual

def main():
    print("Task 1: MLE Convergence with Increasing Data")
    
    true_p = 0.7
    sample_sizes = [10, 30, 50, 100, 500, 1000]
    p_mle_values = []
    
    for n in sample_sizes:
        # Generate varied sizes of coin toss data
        observed_data = data.generate_bernoulli_data(n_samples=n, true_p=true_p, seed=0)
        
        # Calculate maximum likelihood estimate for p
        p_mle = observed_data.mean()
        p_mle_values.append(p_mle)
        
    # Plot convergence of MLE of p as n increases
    visual.plot_mle_convergence(sample_sizes, p_mle_values, true_p)

if __name__ == "__main__":
    main()
