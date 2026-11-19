import numpy as np
import xarray as xr
import glob
import dask
import numcodecs
from zarr.codecs import BloscCodec


def get_chunks(sizes, chunksize=393216):
    dimensions = sizes.keys()
    if ("altitude" in dimensions) and ("time" in dimensions):
        chunks = {
            "time": 2000,
            "altitude": chunksize // 2000,
        }
    elif "time" in dimensions:
        chunks = {
            "time": sizes["time"],
        }
    else:
        chunks = {
            "altitude": sizes["altitude"],
        }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):

    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    return {
        var: {
            "compressors": (compressor,),
            "dtype": dataset[var].dtype,
            "chunks": get_chunks(dataset[var].sizes),
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


async def get_client(**kwargs):
    import aiohttp
    import aiohttp_retry

    retry_options = aiohttp_retry.ExponentialRetry(
        attempts=3, exceptions={OSError, aiohttp.ServerDisconnectedError}
    )
    retry_client = aiohttp_retry.RetryClient(
        raise_for_status=False, retry_options=retry_options
    )
    return retry_client


def bitround(ds, keepbits=16, codec=None):
    def _bitround(var, keepbits, codec=None):
        if codec is None:
            codec = numcodecs.BitRound(keepbits=keepbits)

        return codec.decode(codec.encode(var))

    ds_rounded = xr.apply_ufunc(
        _bitround,
        ds[[var for var in ds.data_vars if np.issubdtype(ds[var].dtype, np.float32)]],
        kwargs={"keepbits": keepbits},
        keep_attrs=True,
        dask="parallelized",
    )

    return xr.merge(
        [
            ds_rounded,
            ds[
                [
                    var
                    for var in ds.data_vars
                    if not np.issubdtype(ds[var].dtype, np.float32)
                ]
            ],
        ]
    )


def get_full_ds_for_var(path, name, ds_var_name):

    var_paths = glob.glob(f"{path}/PERCUSION_*HALO*_WALES_{name}_*.nc")
    alt_grid = np.arange(0, 17000, 15, dtype=np.float32)
    datasets = []
    for var_path in var_paths:
        ds = (
            xr.open_dataset(var_path)
            .chunk(time=2000)
            .sortby("altitude")
            .sel(altitude=slice(0, None))
        )
        print(var_path, flush=True)
        min_alt = ds.altitude.min().values
        index = (np.abs(alt_grid - min_alt)).argmin()
        alt_attrs = ds.altitude.attrs
        datasets.append(
            ds.assign(
                altitude=(
                    "altitude",
                    alt_grid[index : (index + len(ds.altitude))],
                    alt_attrs,
                )
            )
        )
    rename_dict = {ds_var_name: name}

    rename_dict.update(
        {
            key: f"{name}_{key}"
            for key in ["flags", "error", "alpha_to_beta", "off_nadir_angle"]
            if key in datasets[0].variables
        }
    )
    ds = (
        xr.concat(datasets, dim="time")
        .rename(rename_dict)
        .sortby(["time", "altitude"])
        .chunk(time=2000, altitude=-1)
    )
    assert np.all(
        ds.vertical_resolution.max("time") == ds.vertical_resolution.min("time")
    )
    return ds.assign(
        {
            f"{name}_vertical_resolution": (
                "altitude",
                ds.vertical_resolution.mean("time").values,
                ds.vertical_resolution.attrs,
            ),
            f"{name}_horizontal_resolution": ds.horizontal_resolution,
            f"{name}_flags": ds[f"{name}_flags"].astype(np.uint16),
        }
    ).drop_vars(["vertical_resolution", "horizontal_resolution"])
