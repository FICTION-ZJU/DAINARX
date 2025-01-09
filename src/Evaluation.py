import numpy as np

def eva_trace(trace, est_trace, Ts):
    # Extract changepoint sets
    est_chp = est_trace['chpoints']
    chp = trace['chpoints']

    len_chp = min(len(est_chp), len(chp))
    est_chp2 = est_chp[:len_chp]
    chp2 = chp[:len_chp]

    # Compute tau_max
    tau_max = max(abs(np.array(est_chp2) - np.array(chp2)))

    # Compute eps
    eps = 0.0
    for j in range(1, len_chp):
        startj = chp[j - 1]
        endj = chp[j]
        startj_est = est_chp[j - 1]
        endj_est = est_chp[j]

        # Extract the segments
        xs = trace['x'][startj:endj, :]
        xs_est = est_trace['x'][startj_est:endj_est, :]

        # Mitigate ordering effects by executing in both orders
        eps_temp = errorAlignedPoints(xs, xs_est, tau_max)
        eps = max(eps, eps_temp)
        eps_temp = errorAlignedPoints(xs_est, xs, tau_max)
        eps = max(eps, eps_temp)

    return [len(trace['x']) * Ts, len_chp, tau_max * Ts, eps]


def errorAlignedPoints(xs, xs_est, tau_max):
    """
    Computes the maximal distance between matching points from two traces
    allowing for a time shift up to tau_max.

    Parameters:
    xs (ndarray): Original trace segment.
    xs_est (ndarray): Estimated trace segment.
    tau_max (int): Maximum allowed time shift.

    Returns:
    float: Maximum alignment error.
    """
    eps = 0.0
    num_var = xs.shape[1]

    # Go over all points from the original trace
    for h in range(xs.shape[0]):
        # Due to possible time shift, consider matching points from estimated trace
        start_temp = max(h - tau_max, 0)
        end_temp = min(h + tau_max, xs_est.shape[0] - 1)

        # Compute errors for each matching and choose the minimal one
        xs_temp = xs[h, :] - xs_est[start_temp:end_temp + 1, :]
        e_temp = np.min(np.diag(xs_temp @ xs_temp.T))
        eps = max(eps, np.sqrt(e_temp))

    return eps


