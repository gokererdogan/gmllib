"""
gmllib

This file contains classes for loading datasets from disk.
Datasets contain training, validation, and test data, and
all methods in gmllib expect instances of DataSet class to
access data.

Goker Erdogan
2015
https://github.com/gokererdogan
"""
import numpy as np
import os.path

class DataSetException(Exception):
    pass

class MultiFileArray(object):
    """
    This class implements an array data structure for large arrays
    split over multiple files.
    Only accessing elements of the array (either single, or by slicing)
    are allowed.
    Certain slicing constructions are not allowed.
        - End indices cannot be omitted, e.g., x[10:] won't work.
        - Negative slice indices are not allowed, e.g., x[-3:2] won't work.
    """
    def __init__(self, file_list, shape, N_file):
        self.file_list = file_list
        self.shape = shape
        self.ndim = len(self.shape)
        self.N_file = N_file
        # no file is open
        self.open_fid = -1
        self.x = None

        # file id for each index
        self.file_ids = np.zeros(self.shape[0], dtype=int)
        # index into the each separate matrix. map from global to file-dependent indices
        self.indices = np.zeros(self.shape[0], dtype=int)
        s = 0
        for i, n in enumerate(N_file):
            self.indices[s:(s+n)] = np.arange(0, n)
            self.file_ids[s:(s+n)] = i
            s += n

    @staticmethod
    def load(file_list):
        # load the first file
        f = file_list[0]
        x = np.load(f)
        # the shape of the full matrix
        shape = list(x.shape)
        # number of samples per file
        N_file = [shape[0]]

        for f in file_list[1:]:
            del x
            x = np.load(f)
            # we need to make sure the number of variables (columns) are the same across files
            if list(x.shape[1:]) != shape[1:]:
                raise DataSetException("Number of variables (columns) should be the same across files.")
                return None
            N = x.shape[0]
            shape[0] += N
            N_file.append(N)

        return MultiFileArray(file_list, shape, N_file)

    def _load_file(self, fid):
        if self.x is not None:
            del self.x
        self.x = np.load(self.file_list[fid])
        self.open_fid = fid

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0 or item > self.shape[0]:
                raise IndexError("Index out of bounds.")

            # check if data is already available
            fid = self.file_ids[item]
            if fid != self.open_fid:
                self._load_file(fid)

            return self.x[self.indices[item]]

        elif isinstance(item, slice):
            if item.step is None:
                step = 1

            if item.start < 0 or item.stop > self.shape[0]:
                raise IndexError("Index out of bounds.")

            i = np.arange(item.start, item.stop, step)

            # create array for returning data
            data_shape = [i.size, ]
            data_shape.extend(self.shape[1:])
            data = np.zeros(data_shape)

            # get ids of files we need
            fids = list(np.unique(self.file_ids[i]))
            # if one of the files we need is already open

            if self.open_fid in fids:
                data[self.file_ids[i] == self.open_fid] = self.x[self.indices[i[self.file_ids[i] == self.open_fid]]]
                fids.remove(self.open_fid)

            # read the remaining data
            for fid in fids:
                self._load_file(fid)
                data[self.file_ids[i] == fid] = self.x[self.indices[i[self.file_ids[i] == fid]]]

            return data
        else:
            raise DataSetException("MultiFileArray can only be indexed with int or slices.")

class Data(object):
    """
    A generic Data class for training, validation or test
    data. Handles both supervised and unsupervised data.
    """
    def __init__(self, x, y=None):
        if y is not None and x.shape[0] != y.shape[0]:
            raise DataSetException("Number of samples and labels should be the same.")

        self._x = x
        self._y = y
        self.N = self._x.shape[0]
        self.D = 1
        if self._x.ndim != 1:
            self.D = self._x.shape[1]
        if self._y is None:
            self.supervised = False
        else:
            self.supervised = True
            self.K = 1
            if self._y.ndim != 1:
                self.K = self._y.shape[1]

    @staticmethod
    def load(xfile, yfile=None):
        # if xfile or yfile are list of filenames, use MultiFileArray
        if isinstance(xfile, list):
            x = MultiFileArray.load(xfile)
        else:
            x = np.load(xfile)
        y = None
        if yfile is not None:
            if isinstance(yfile, list):
                y = MultiFileArray.load(yfile)
            else:
                y = np.load(yfile)
        return Data(x, y)

    def shuffle(self):
        pass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        raise DataSetException("Data cannot be updated after initialization.")

    @property
    def y(self):
        if not self.supervised:
            raise DataSetException("No labels are available.")
        return self._y

    @y.setter
    def y(self, value):
        raise DataSetException("Data cannot be updated after initialization.")

    def __getstate__(self):
        """
        If for any reason someone wants to pickle a Data instance, we don't
        want to pickle the data array itself.
        """
        return {k: v for k, v in self.__dict__.iteritems() if k not in ['x', 'y', '_x', '_y']}


class DataSet(object):
    """
    DataSet class
    """
    def __init__(self, name, trainx, trainy=None, validationx=None, validationy=None, testx=None, testy=None):
        self.name = name
        self.train = Data.load(trainx, trainy)
        self.D = self.train.D
        self.supervised = self.train.supervised
        if self.supervised:
            self.K = self.train.K

        if validationx is not None:
            self.validation = Data.load(validationx, validationy)
            if self.validation.D != self.D:
                raise DataSetException("Training and validation sets do not have the same number of inputs.")
            if self.validation.K != self.K:
                raise DataSetException("Training and validation sets do not have the same number of outputs.")
            if self.validation.supervised != self.supervised:
                raise DataSetException("You can't have labels for one set while the other does not.")

        if testx is not None:
            self.test = Data.load(testx, testy)
            if self.test.D != self.D:
                raise DataSetException("Training and test sets do not have the same number of inputs.")
            if self.test.supervised and self.test.K != self.K:
                raise DataSetException("Training and test sets do not have the same number of outputs.")

    @staticmethod
    def load_from_path(name, folder):
        """
        Load dataset from files in a given folder.
        This function looks for files
            train_x.npy, train_y.npy, val_x.npy, val_y.npy, test_x.npy and test_y.npy
        Returns a DataSet instance
        """
        tx_file = "{0:s}/train_x.npy".format(folder)
        if not os.path.isfile(tx_file):
            raise DataSetException("Training file not found.")

        ty_file = "{0:s}/train_y.npy".format(folder)
        if not os.path.isfile(ty_file):
            ty_file = None

        vx_file = "{0:s}/val_x.npy".format(folder)
        if not os.path.isfile(vx_file):
            vx_file = None

        vy_file = "{0:s}/val_y.npy".format(folder)
        if not os.path.isfile(vy_file):
            vy_file = None

        sx_file = "{0:s}/test_x.npy".format(folder)
        if not os.path.isfile(sx_file):
            sx_file = None

        sy_file = "{0:s}/test_y.npy".format(folder)
        if not os.path.isfile(sy_file):
            sy_file = None

        return DataSet(name=name, trainx=tx_file, trainy=ty_file, validationx=vx_file, validationy=vy_file,
                       testx=sx_file, testy=sy_file)

