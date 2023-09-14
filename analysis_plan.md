Analysis Plan:

1. Make a brain mask for each volume using threshold 
    ```
    T = np.mean(vol) / 8
    return vol[vol > T]
    ```
2. Average all masks and threshold by 0.5
3. Removing effects of covariates (like task)
4. Calculate pair-wise DVARS for each combination of volumes
5. Normalize all DVARS by z-score
6. Threshold the z-score based on standard deviation - (change threshold to make it 
more or less sensitive based on visual inspection)