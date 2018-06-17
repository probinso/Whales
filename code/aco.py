import datetime
from datetime import datetime, timedelta
import os.path as osp
import re

from PyEMD import EMD
from memoized_property import memoized_property

import numpy as np
import scipy.signal as signal

import warnings

from operator import attrgetter

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html

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
        # 2016-02-15--05.00.HYD24BBpk
        name = osp.basename(filename)
        dts, encs = name.rsplit('.', 1)
        time_stamp = datetime.strptime(dts, cls.time_code)

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
    res = timedelta(minutes=5)

    @classmethod
    def __floor_dt(cls, dt):
        src = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        offset = src.total_seconds() % cls.res.total_seconds()
        return dt - timedelta(seconds=offset)

    @classmethod
    def _filename_from_date(cls, index_datetime):
        dts = datetime.strftime(index_datetime, cls.time_code)
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
        if isinstance(target, datetime):
            return _DatetimeACOLoader.load_ACO_from_datetime(self.basedir, target)

from collections import namedtuple

PlotInfo = namedtuple('PlotInfo', ['data', 'xaxis', 'interval', 'shift'])

class ACO:
    def __init__(self, time_stamp, fs, data, centered=False):
        self._time_stamp = time_stamp
        self._fs = fs
        self._data = data.astype(np.float64)
        self._centered = centered

    @memoized_property
    def _max_value(self):
        return np.max(np.abs(self._data))

    @memoized_property
    def normdata(self, dtype=np.int32):
        data = self._data.copy()
        max_value = self._max_value
        data = ((data/max_value) * np.iinfo(dtype).max).astype(dtype)
        return data

    def resample(self, n):
        if len(self) == n:
            return self.copy()

        fs_ratio = n/len(self._data)
        warnings.warn(f'Only {fs_ratio:.3f} of signal represented', UserWarning)
        x = signal.resample(self._data, n)
        return ACO(
            self._time_stamp,
            int(np.round(self._fs * fs_ratio)),
            x
        )

    def _resample_fs(self, fs):
        fs_ratio = fs/self._fs
        data = signal.resample(self._data, int(np.round(len(self)*fs_ratio)))
        return data

    @memoized_property
    def _emd(self):
        emd = EMD()
        return emd(self._data)

    def remove_dc(self, levels=1):
        assert(levels != 0)
        IMFs = self._emd
        return ACO(
            self._time_stamp,
            self._fs,
            self._data - np.sum(IMFs[levels:], axis=0),
            True
        )

    def sloppy_remove_dc(self):
        warnings.warn('Do not use in production. No justification for method.', UserWarning)
        n = len(self)
        x = signal.resample(signal.resample(self._data, 1000), n)

        return ACO(
            self._time_stamp,
            self._fs,
            self._data - x,
            True
        )

    def Listen(self, data=None):
        if data is None:
            data = self._data.copy()

        # bug in IPython.Audio, only handles common fs
        fs = 24000
        data = self._resample_fs(24000)

        from IPython.display import Audio
        return Audio(data=data, rate=fs)

    def spectrogram(self, frame_duration=.08, frame_shift=.001, wtype='hanning'):
        unit = self._Frame(frame_duration, frame_shift)
        mat = unit.data * signal.get_window(wtype, unit.data.shape[1])
        N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
        return unit._replace(data=np.fft.rfft(mat, n=N))

    def logspectrogram(self, frame_duration=.08, frame_shift=.001, wtype='hanning'):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(20 * np.log10(np.abs(unit.data))))

    def autocorr(self):
        x = self._data
        n = len(x)
        return np.correlate(x, x, mode='full')[n - 1:]

    def periodogram(self):#, frame_duration=.08, frame_shift=.001, wtype='rectagle'):
        return signal.periodogram(self._data, fs=self._fs)

    def cepstrum(self, frame_duration=.08, frame_shift=.001, wtype='hanning'):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(np.fft.irfft(np.log(np.abs(unit.data))).real))

    def View(self, itype=None, **kwargs):
        if itype is None:
            unit = self._data
        elif hasattr(self, itype):
            attr = getattr(self, itype)
            unit = attr(**kwargs) if callable(attr) else attr
        else:
            raise "Fuck You"

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        _ = plt.title(itype)

        if isinstance(unit, PlotInfo):
            '''
            ['data', 'xaxis', 'yaxis'])
            _ = plt.plot(unit.data.T.real)
            '''

            _ = plt.imshow(X=unit.data.T.real, interpolation=None)
            _ = plt.yticks([])
            _ = plt.ylabel(f'{unit.interval:.3f} interval, {unit.shift:.3f} shift, {self._fs} f/s')

            #_ = plt.xticks(unit.xaxis) # too large
        elif len(unit.shape) == 1:
            _ = plt.plot(unit)
        elif len(unit.shape) == 2:
            _ = plt.imshow(X=unit.T.real, interpolation=None)
        else:
            raise "DUM DUM DUM"

    def __len__(self):
        return len(self._data)

    def _Frame(self, frame_duration=.08, frame_shift=.001):
        toint = lambda f: int(np.round(f))

        n = toint(self._fs * frame_duration)
        s = toint(self._fs * frame_shift)

        total_frames = (len(self._data) - n) // s + 1
        time = (self._time_stamp + (timedelta(seconds=frame_shift) * i)
                for i in range(total_frames))

        dom = np.arange(total_frames) * s + n // 2
        mat = np.empty((total_frames, n))
        mat[:,:] = np.NAN

        start = 0
        for i in range(total_frames):
            idx = slice(start, (start+n))
            mat[i, :] = self._data[idx]
            start += s
        return PlotInfo(mat, time, frame_duration, frame_shift)

    def delta_offset(self, t):
        return int(t.total_seconds() * self._fs)

    def _date_offset(self, d):
        return self.delta_offset(d - self._time_stamp)

    def __getitem__(self, slice_):
        i, j = slice_.start, slice_.stop

        if None in [i, j]:
            raise "Needs Testing"

        new_start \
            = timedelta(0) if i is None else i

        new_end \
            = self._durration if j is None else j


        if new_start.total_seconds() < 0:
            raise "PreIndexError"

        if new_end.total_seconds() > self._durration.total_seconds():
            raise "PostIndexError"

        return ACO(
            self._time_stamp + new_start,
            self._fs,
            self._data[self.delta_offset(new_start):
                       self.delta_offset(new_end)]
        )

    def copy(self):
        return ACO(
            self._time_stamp,
            self._fs,
            self._data.copy()
        )

    @property
    def end_datetime(self):
        return self._time_stamp + self._durration

    @memoized_property
    def _durration(self):
        return timedelta(seconds=float((self._data.size / self._fs)))

    def __matmul__(self, other):
        assert(self._centered == other._centered)

        ordered = (self, other) # wlg
        if self._fs != other._fs:
            ordered = sorted((self, other), key=attrgetter('_fs'))
            ordered[-1] = ACO(
                ordered[-1]._time_stamp,
                ordered[0]._fs,
                ordered[-1]._resample_fs(ordered[0]._fs)
            )

        ordered = sorted(ordered, key=attrgetter('_time_stamp'))
        end = max(map(attrgetter('end_datetime'), ordered))
        date_offset = ordered[0]._date_offset
        space = date_offset(end)

        data = np.full(space, np.NAN)
        data[:len(ordered[0])] = ordered[0]._data

        start = date_offset(ordered[-1]._time_stamp)
        idx = slice(start, start+len(ordered[-1]))
        overlap_count = np.sum(ordered[-1]._data[np.where(data[idx] is not np.NAN)] is not np.NAN)

        if overlap_count > 0:
            warnings.warn(f'Overlaps {overlap_count} samples', UserWarning)

        data[idx] = ordered[-1]._data

        return ACO(ordered[0]._time_stamp,
                   ordered[0]._fs,
                   data,
                   ordered[0]._centered
        )
