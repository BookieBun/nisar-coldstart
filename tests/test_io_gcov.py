"""Reader tests on a tiny HDF5 file laid out like a NISAR GCOV product, read both from disk and
through HTTPRangeFile with a fake session that serves byte ranges of that file. No network."""
import h5py
import numpy as np
import pytest

from coldstart.io_gcov import GRID_GROUP, HTTPRangeFile, _GCOVFile, chunk_ranges, open_gcov, read_grid_info, read_window

H, W = 96, 128


@pytest.fixture
def gcov_file(tmp_path):
    rng = np.random.default_rng(0)
    data = {"HHHH": rng.gamma(4, 0.02, (H, W)).astype(np.float32),
            "HVHV": rng.gamma(4, 0.005, (H, W)).astype(np.float32),
            "rtcGammaToSigmaFactor": rng.uniform(0.5, 1, (H, W)).astype(np.float32),
            "numberOfLooks": np.full((H, W), 4, np.float32),
            "mask": np.ones((H, W), np.uint8)}
    p = tmp_path / "NISAR_L2_PR_GCOV_test.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group(GRID_GROUP.format(freq="A"))
        for k, v in data.items():
            g.create_dataset(k, data=v, chunks=(32, 32), compression="gzip", compression_opts=1)
        g["xCoordinates"] = 324005.0 + 10.0 * np.arange(W)        # pixel centres
        g["yCoordinates"] = 5101915.0 - 10.0 * np.arange(H)
        g.create_dataset("projection", data=np.uint32(32611)).attrs["epsg_code"] = 32611
    return p, data


class FakeResponse:
    def __init__(self, status, content=b"", total=0, url="https://signed.example/file.h5", start=0):
        self.status_code, self.content, self.url = status, content, url
        self.headers = {"Content-Range": f"bytes {start}-{start + len(content) - 1}/{total}"}


class FakeSession:
    """Serves ranges of a local file; fails according to a script of statuses or exceptions."""

    def __init__(self, path, failures=()):
        self.blob = path.read_bytes()
        self.failures = list(failures)
        self.calls = 0

    def get(self, url, headers=None, **kw):
        self.calls += 1
        a, b = (int(x) for x in headers["Range"].split("=")[1].split("-"))
        if self.calls > 1 and self.failures:        # never fail the opening probe
            f = self.failures.pop(0)
            if isinstance(f, Exception):
                raise f
            return FakeResponse(f)
        return FakeResponse(206, self.blob[a:b + 1], len(self.blob), start=a)


def test_grid_info_and_window_from_disk(gcov_file):
    p, data = gcov_file
    with open_gcov(p) as f:
        gi = read_grid_info(f)
        assert gi.crs_epsg == 32611 and gi.spacing_m == 10.0 and gi.shape == (H, W)
        assert gi.transform == (10.0, 0.0, 324000.0, 0.0, -10.0, 5101920.0)   # corner, not centre
        lay = read_window(f, (10, 50, 20, 100))
    assert set(lay) == {"HHHH", "HVHV", "mask", "numberOfLooks", "rtcGammaToSigmaFactor"}
    assert np.array_equal(lay["HVHV"], data["HVHV"][10:50, 20:100]) and lay["HHHH"].dtype == np.float32


def test_missing_layer_or_coordinates_raise(gcov_file):
    p, _ = gcov_file
    with h5py.File(p, "a") as f:
        del f[GRID_GROUP.format(freq="A")]["HVHV"]
        del f[GRID_GROUP.format(freq="A")]["xCoordinates"]
    with open_gcov(p) as f:
        with pytest.raises(KeyError, match="HVHV"):
            read_window(f, (0, 8, 0, 8))
        with pytest.raises(KeyError, match="xCoordinates"):
            read_grid_info(f)


def _open_stream(path, **kw):
    raw = HTTPRangeFile("https://example/file.h5", session=FakeSession(path, **kw), block_size=4096)
    raw.retry_wait = 0.0
    f = _GCOVFile(raw, "r")
    f._coldstart_raw = raw
    return f, raw


def test_streamed_window_equals_disk_with_prefetch(gcov_file):
    p, data = gcov_file
    f, raw = _open_stream(p)
    with f:
        ds = f[GRID_GROUP.format(freq="A")]["HHHH"]
        assert len(chunk_ranges(ds, (10, 50, 20, 100))) == 2 * 4           # rows 0-63, cols 0-127 in 32 px chunks
        lay = read_window(f, (10, 50, 20, 100), layers=["HHHH", "mask"], workers=4)
    assert np.array_equal(lay["HHHH"], data["HHHH"][10:50, 20:100])
    assert raw.bytes_fetched < raw.size * 3 and not raw._ranges              # prefetch buffer released


def test_stream_retries_transient_failures(gcov_file):
    p, data = gcov_file
    f, raw = _open_stream(p, failures=[ConnectionError("dropped"), 503, 429])
    with f:
        lay = read_window(f, (0, 32, 0, 32), layers=["HVHV"])
    assert np.array_equal(lay["HVHV"], data["HVHV"][:32, :32])


def test_stream_gives_up_with_clear_error(gcov_file):
    p, _ = gcov_file
    raw = HTTPRangeFile("https://example/file.h5", session=FakeSession(p, failures=[404]), block_size=4096)
    raw.retry_wait = 0.0
    with pytest.raises(OSError, match="HTTP 404"):
        raw._get(0, 99)
    raw2 = HTTPRangeFile("https://example/file.h5", session=FakeSession(p, failures=[TimeoutError("slow")] * 9), block_size=4096)
    raw2.retry_wait = 0.0
    with pytest.raises(OSError, match="after 4 tries"):
        raw2._get(0, 99)
