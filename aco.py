import datetime
import os.path as osp
import re

import numpy as np
import scipy.signal as signal

class _ACOLoader:
    header_dtype = np.dtype(
        [('Record', '<u4'),
         ('Decimation', '<u1'),
         ('StartofFile', '<u1'),
         ('Sync1', '<u1'),
         ('Sync2', '<u1'),
         ('Statusbyte1', '<u1'),
         ('Statusbyte2', '<u1'),
         ('pad1', '<u1'),
         ('LeftRightFlag', '<u1'),
         ('tSec', '<u4'),
         ('tuSec', '<u4'),
         ('timecount', '<u4'),
         ('Year', '<i2'),
         ('yDay', '<i2'),
         ('Hour', '<u1'),
         ('Min', '<u1'),
         ('Sec', '<u1'),
         ('Allignment', '<u1'),
         ('sSec', '<i2'),
         ('dynrange', '<u1'),
         ('bits', '<u1')])

    resolution = np.int32
    time_code = '%Y-%m-%d--%H.%M'

    @classmethod
    def load_ACO_from_file(cls, filename):
        time_stamp, fs = cls._params_from_filename(filename)
        data = cls._from_file(filename)
        return ACO(time_stamp, fs, data)

    @classmethod
    def _ACO_to_int(cls, databytes, nbits):
        """
        Convert the block of bytes to an array of int32.

        We need to use int32 because there can be 17 bits.
        """
        nbits = int(nbits)
        # Fast path for special case of 16 bits:
        if nbits == 16:
            return databytes.view(np.int16).astype(cls.resolution)
        # Put the bits in order from LSB to MSB:
        bits = np.unpackbits(databytes).reshape(-1, 8)[:, ::-1]
        # Group by the number of bits in the int:
        bits = bits.reshape(-1, nbits)
        # Reassemble the integers:
        pows = 2 ** np.arange(nbits, dtype=cls.resolution)
        num = (bits * pows).sum(axis=1).astype(cls.resolution)
        # Handle twos-complement negative integers:
        neg = num >= 2**(nbits-1)
        num[neg] -= 2**nbits
        return num

    @classmethod
    def _params_from_filename(cls, filename):
        #2016-02-15--05.00.HYD24BBpk
        name = osp.basename(filename)
        dts, encs = name.rsplit('.', 1)
        time_stamp = datetime.datetime.strptime(dts, cls.time_code)

        fs = int(re.findall('\d+', encs).pop())*1000
        return time_stamp, fs

    @classmethod
    def _from_file(cls, filename):
        headerlist = []
        datalist = []
        with open(filename, 'rb') as fid:
            fid.seek(0, 2)
            eof = fid.tell()
            fid.seek(0, 0)
            while fid.tell() < eof:
                header = np.fromfile(fid, count=1, dtype=cls.header_dtype)[0]
                headerlist.append(header)
                nbits = int(header['bits'])
                count = (4096//8) * nbits
                databytes = np.fromfile(fid, count=count, dtype='<u1')
                data = cls._ACO_to_int(databytes, nbits)
                datalist.append(data)

        headers = np.array(headerlist)

        # Keeping the blocks separate, matching the headers:
        data = np.vstack(datalist)

        # But we can also view it as a single time series:
        alldata = data.reshape(-1)
        return alldata

class _DatetimeACOLoader(_ACOLoader):
    res = datetime.timedelta(minutes=5)

    @classmethod
    def __floor_dt(cls, dt):
        src = datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        offset = src.total_seconds() % cls.res.total_seconds()
        return dt - datetime.timedelta(seconds=offset)

    @classmethod
    def _filename_from_date(cls, index_datetime):
        dts = datetime.datetime.strftime(index_datetime, cls.time_code)
        encs = 'HYD24BBpk'
        return '.'.join([dts, encs])

    @classmethod
    def _path_from_date(cls, index_datetime):
        info = [index_datetime.year, index_datetime.month, index_datetime.day]
        dirname = osp.join(*map(lambda i: str(i).zfill(2), info))
        basename = cls._filename_from_date(index_datetime)
        return osp.join(dirname, basename)

    @classmethod
    def load_ACO_from_datetime(cls, storage_dir, index_datetime):
        floor_datetime = cls.__floor_dt(index_datetime)
        fullpath = osp.join(storage_dir, cls._path_from_date(floor_datetime))
        return cls.load_ACO_from_file(fullpath)

class ACOio:
    def __init__(self, basedir):
        self.basedir = basedir

    def load(self, target):
        if isinstance(target, str):
            return _ACOLoader.load_ACO_from_file(target)
        if isinstance(target, datetime.datetime):
            return _DatetimeACOLoader.load_ACO_from_datetime(self.basedir, target)

class ACO:
    def __init__(self, time_stamp, fs, data):
        self._time_stamp = time_stamp
        self._fs = fs
        self._data = data.astype(np.float64)

    @property
    def _max_value(self):
        return np.max(np.abs(self._data))

    @property
    def normdata(self, dtype=np.int32):
        data = self._data.copy()
        max_value = self._max_value
        data = ((data/max_value) * np.iinfo(dtype).max).astype(dtype)
        return data

    def Listen(self):
        from IPython.display import Audio
        return Audio(data=self.normdata, rate=self._fs)

    def spectrogram(self, frame_duration=.008, frame_shift=.0065, wtype='hanning'):
        mat = self._Frame(frame_duration, frame_shift)
        mat *= signal.get_window(wtype, mat.shape[1])
        N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
        return np.fft.rfft(mat, n=N)

    def logspectrogram(self, frame_duration=.008, frame_shift=.0065, wtype='hanning'):
        mat = self.spectrogram(frame_duration, frame_shift, wtype)
        return 20 * np.log10(np.abs(mat))

    def autocorrelogram(self, frame_duration=.008, frame_shift=.0065, wtype='hanning'):
        mat = self._data
        return  np.correlate(mat, mat, mode='same')

    def cepstrum(self, frame_duration=.008, frame_shift=.0065, wtype='hanning'):
        mat = self.spectrogram(frame_duration, frame_shift, wtype)
        return np.fft.irfft(np.log(np.abs(mat))).real

    def View(self, itype=None, **kwargs):
        if itype is None:
            data = self._data
        elif hasattr(self, itype):
            attr = getattr(self, itype)
            data = attr(**kwargs) if callable(attr) else attr
        else:
            raise "Fuck You"

        from matplotlib import pyplot as plt
        fig = plt.figure()

        plt.title(str(dict(max=data.max(), min=data.min(), shape=data.shape)))
        if len(data.shape) == 1:
            _ = plt.plot(data)
        elif len(data.shape) == 2:
            _ = plt.imshow(X=data.T.real, interpolation=None)
        else:
            raise "DUM DUM DUM"

    def _Frame(self, frame_duration=.008, frame_shift=.0065):
        toint = lambda f: int(np.round(f))

        n = toint(self._fs * frame_duration)
        s = toint(self._fs * frame_shift)

        total_frames = (len(self._data) - n) // s + 1

        dom = np.arange(total_frames) * s + n // 2
        mat = np.empty((total_frames, n))
        mat[:,:] = np.NAN

        start = 0
        for i in range(total_frames):
            idx = slice(start, (start+n))
            mat[i, :] = self._data[idx]
            start += s
        return mat

    def time_offset(self, t):
        # XXX if t is None? fix results
        return (t - self._time_stamp)

    def frame_offset(self, t):
        if t is None:
            return None
        return int(self.time_offset(t).total_seconds()) * self._fs

    def __getitem__(self, slice_):
        i, j = slice_.start, slice_.stop

        return ACO(
            self._time_stamp + (0 if i is None else self.time_offset(i)),
            self._fs,
            self._data[self.frame_offset(i):self.frame_offset(j)].copy(),
        )

    @property
    def durration(self):
        return (self._data.size // self._fs) // 60

    def __matmul__(self, other):
        pass
