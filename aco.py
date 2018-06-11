import datetime
import os.path as osp
import re

import numpy as np

class ACOLoader:
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

    def __init__(self, time_stamp=None, fs=None, data=None, max_value=None, initialized=False):
        self._time_stamp = time_stamp
        self._fs = fs
        self._data = data
        self._max_value = max_value
        self._initialized = initialized

    def load_from_ACO_file(self, filename):
        self._time_stamp, self._fs = self._params_from_filename(filename)
        self._data, self._max_value = self._from_file(filename)
        self._initialized = True

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
    def __floor_dt(cls, dt, *, res=datetime.timedelta(minutes=5)):
        src = datetime.timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        offset = src.total_seconds() % res.total_seconds()
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

    def load_from_date(self, storage_dir, index_datetime):
        floor_datetime = self.__floor_dt(index_datetime)
        fullpath = osp.join(storage_dir, self._path_from_date(floor_datetime))
        self.load_from_ACO_file(fullpath)
        self._initialized = True

    @classmethod
    def _params_from_filename(cls, filename):
        #2016-02-15--05.00.HYD24BBpk
        name = osp.basename(filename)
        dts, encs = name.rsplit('.', 1)
        time_stamp = datetime.datetime.strptime(dts, cls.time_code)

        fs = int(re.findall('\d+', encs).pop())*1000
        return time_stamp, fs

    def _from_file(self, filename):
        headerlist = []
        datalist = []
        with open(filename, 'rb') as fid:
            fid.seek(0, 2)
            eof = fid.tell()
            fid.seek(0, 0)
            while fid.tell() < eof:
                header = np.fromfile(fid, count=1, dtype=self.header_dtype)[0]
                headerlist.append(header)
                nbits = int(header['bits'])
                count = (4096//8) * nbits
                databytes = np.fromfile(fid, count=count, dtype='<u1')
                data = self._ACO_to_int(databytes, nbits)
                datalist.append(data)

        headers = np.array(headerlist)

        # Keeping the blocks separate, matching the headers:
        data = np.vstack(datalist)

        # But we can also view it as a single time series:
        alldata = data.reshape(-1)
        max_value = np.max(np.abs(alldata))
        return alldata, max_value


class ACO(ACOLoader):
    @property
    def normdata(self):
        data = self._data.copy()
        max_value = self._max_value
        view = self.resolution
        data = ((data.astype(np.float64)/max_value) * np.iinfo(view).max).astype(view)
        return data

    def Listen(self, ):
        from IPython.display import Audio
        return Audio(data=self.normdata, rate=self._fs)

    def View(self):
        pass

    def time_offset(self, t):
        return (t - self._time_stamp)

    def frame_offset(self, t):
        return int(self.time_offset(t).total_seconds()) * self._fs

    def __getitem__(self, slice_):
        i, j = slice_.start, slice_.stop

        result = ACO(
            self._time_stamp+self.time_offset(i),
            self._fs,
            self._data[self.frame_offset(i):self.frame_offset(j)].copy(),
            self._max_value,
            True
        )
        return result
