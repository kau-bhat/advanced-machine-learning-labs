import numpy as np
from scipy.stats import norm
import data
import visual

def main():
    print("Task 0: Histogram of Data with Fitted Normal Distribution")
    
    # 1. Provide electricity usage array
    observed_data = data.get_electricity_usage_data()
    
    # 2. Calculate the sample mean and variance of the data
    mean = np.mean(observed_data)
    variance = np.var(observed_data, ddof=1)
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}") 
    
    # 3. Create probability distribution array
    x = np.linspace(min(observed_data)-1, max(observed_data)+1, 100)
    pdf = norm.pdf(x, mean, np.sqrt(variance))
    
    # 4. Use predefined viewing tools
    visual.plot_histogram_normal_fit(observed_data, x, pdf)

if __name__ == "__main__":
    main()
