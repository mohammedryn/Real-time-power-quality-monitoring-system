import numpy as np
from scipy.stats import vonmises

def generate_sample(class_id: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 100ms (500 sample) window of voltage and current for a specific PQ class.
    Returns: (v_sig, i_sig) as float64 physical arrays.
    """
    FS = 5000
    N = 500
    t = np.linspace(0, N/FS, N, endpoint=False)
    
    # --- 1. Common Fundamental Generation ---
    V1 = rng.uniform(200, 245) * np.sqrt(2)
    phi1 = rng.uniform(0, 2 * np.pi)
    I1 = rng.uniform(1.0, 15.0)
    
    # Default to resistive for faults unless overridden
    dpf = rng.uniform(0.85, 1.0) if class_id in [0, 6] else 1.0 
    phi_i = phi1 - np.arccos(dpf)
    
    v_sig = V1 * np.sin(2 * np.pi * 50 * t + phi1)
    i_sig = I1 * np.sin(2 * np.pi * 50 * t + phi_i)

    # --- 2. Class-Specific Modulations ---
    if class_id == 0:  # Normal
        thd_bg = rng.uniform(0.01, 0.04)
        for h in [3, 5, 7]:
            amp = (thd_bg / np.sqrt(3)) * V1 * rng.uniform(0.5, 1.5)
            v_sig += amp * np.sin(2 * np.pi * h * 50 * t + rng.uniform(0, 2 * np.pi))

    elif class_id == 1:  # Sag
        depth = rng.uniform(0.1, 0.9)
        duration = rng.uniform(0.5, 30.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_sig[mask] *= (1.0 - depth)
        i_sig[mask] *= (1.0 - depth)

    elif class_id == 2:  # Swell
        depth = float(rng.beta(1.5, 5.0)) * 0.7
        duration = rng.uniform(0.5, 30.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_sig[mask] *= (1.0 + depth)
        i_sig[mask] *= (1.0 + depth)

    elif class_id == 3:  # Interruption
        residual = rng.uniform(0.0, 0.1)
        duration = rng.uniform(1.0, 5.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_sig[mask] *= residual
        i_sig[mask] *= residual

    elif class_id == 4:  # Harmonic Distortion
        subtype = rng.integers(0, 3)
        if subtype == 0:   # SMPS
            phi3 = vonmises.rvs(kappa=3.0, loc=np.pi/4, random_state=rng)
            phi5 = vonmises.rvs(kappa=2.5, loc=-np.pi/5, random_state=rng)
            phi7 = vonmises.rvs(kappa=2.0, loc=np.pi/7, random_state=rng)
            r3, r5, r7 = rng.uniform(0.30, 0.90), rng.uniform(0.10, 0.40), rng.uniform(0.05, 0.20)
            
            v_sig += (r3 * 0.05 * V1) * np.sin(2 * np.pi * 150 * t + phi3) + \
                     (r5 * 0.02 * V1) * np.sin(2 * np.pi * 250 * t + phi5)
            i_sig += (r3 * I1) * np.sin(2 * np.pi * 150 * t + phi3) + \
                     (r5 * I1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * I1 * 0.5) * np.sin(2 * np.pi * 350 * t + phi7)
                     
        elif subtype == 1:  # VFD
            phi5 = vonmises.rvs(kappa=4.0, loc=-2*np.pi/5, random_state=rng)
            phi7 = vonmises.rvs(kappa=4.0, loc=2*np.pi/7, random_state=rng)
            r5, r7 = rng.uniform(0.15, 0.25), rng.uniform(0.08, 0.15)
            
            v_sig += (r5 * 0.03 * V1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * 0.02 * V1) * np.sin(2 * np.pi * 350 * t + phi7)
            i_sig += (r5 * I1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * I1) * np.sin(2 * np.pi * 350 * t + phi7)
                     
        else:  # Transformer Saturation
            phi3 = vonmises.rvs(kappa=3.5, loc=np.pi/2, random_state=rng)
            r3 = rng.uniform(0.05, 0.20)
            v_sig += (r3 * V1) * np.sin(2 * np.pi * 150 * t + phi3)
            i_sig += (r3 * I1 * 0.8) * np.sin(2 * np.pi * 150 * t + phi3)

    elif class_id == 5:  # Transient
        A_t = rng.uniform(0.1, 1.5) * V1
        tau = rng.uniform(0.1, 2.0) / 1000.0
        f_t = rng.uniform(500, 2000)
        t_start = rng.uniform(0.0, (N/FS) - 0.005)
        polarity = rng.choice([-1, 1])
        
        t_rel = t - t_start
        
        # --- FIXED LOGIC ---
        # Initialize an array of zeros, then apply math ONLY to the valid active window
        transient = np.zeros_like(t)
        mask = t_rel >= 0
        transient[mask] = polarity * A_t * np.exp(-t_rel[mask] / tau) * np.sin(2 * np.pi * f_t * t_rel[mask])
        # -------------------
        
        v_sig += transient


    elif class_id == 6:  # Flicker
        m = rng.uniform(0.05, 0.20)
        f_f = rng.uniform(1.0, 25.0)
        mod_envelope = (1 + m * np.sin(2 * np.pi * f_f * t))
        
        v_sig = V1 * mod_envelope * np.sin(2 * np.pi * 50 * t + phi1)
        i_sig = I1 * mod_envelope * np.sin(2 * np.pi * 50 * t + phi_i)

    # --- 3. AWGN Injection ---
    v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
    i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)

    return v_sig, i_sig