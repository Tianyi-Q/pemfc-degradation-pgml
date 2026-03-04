# since experimental data is time-consuming to get, this script generates synthetic pemfc operating data. 
# I have yet to test all of this on actual .fcd results, but that's on my to-do list, meaning that I will get to it...probably:)

import numpy as np
import pandas as pd
import os

def generate_synthetic_pemfc_data(output_path="data/raw/synthetic_matrix.csv"):
    loadings = [0.05, 0.1, 0.2, 0.3] 
    rhs = [30, 60, 80, 100] # the loadings and RHs are from an earlier experimental study, see GDL project on my site https://tianyi-q.github.io/
    hours = np.linspace(0, 500, 500) # 1-hour resolution
    
    data = []
    
    for load in loadings:
        for rh in rhs:
            # base voltage and degradation kinetics dependent on parameters. 
            # a bit simplified but works, the trend should be correct
            v_init = 0.95 - (0.05 * (100 - rh) / 100) 
            decay_rate = 0.0001 * (0.4 - load) * (110 - rh)
            
            for t in hours:
                #eExponential decay: V(t) = V_init * e^(-decay * t)
                v_t = v_init * np.exp(-decay_rate * t)
                
                # noise
                noise = np.random.normal(0, 0.002)
                
                # flooding spikes (especially high RH)
                flooding = 0
                if rh >= 80 and np.random.rand() > 0.98:
                    flooding = np.random.uniform(-0.05, -0.15)
                    
                v_final = v_t + noise + flooding
                # i added random noise and spiking since the performance can vary A LOT. 
                # try running a cell at 30RH and you'll see a roller coaster
                data.append([load, rh, t, v_final])
                
    df = pd.DataFrame(data, columns=['TiO2_Loading', 'RH_Percent', 'Time_Hours', 'Voltage'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[*] Synthetic 4x4 matrix saved to {output_path} with {len(df)} records.")

if __name__ == "__main__":
    generate_synthetic_pemfc_data()