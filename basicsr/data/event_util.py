import numpy as np


def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    Args:
        events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity].
        num_bins: the number of bins in the temporal axis of the voxel grid.
        width, height: dimensions of the voxel grid.
        return_format: 'CHW' or 'HWC'.

    Returns:
        np.array: The voxel grid
    """

    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if events.size != 0:
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]

        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT  #
        ts = events[:, 0]
        xs = events[:, 1].astype(np.int32)
        ys = events[:, 2].astype(np.int32)
        pols = events[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int32)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < num_bins  # [True True ... True]

        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width  ## ! ! !
                + tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
                + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])
        
    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1, 2, 0)