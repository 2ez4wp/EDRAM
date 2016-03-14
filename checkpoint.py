from __future__ import division, print_function

import cPickle as pickle
import sys
from cStringIO import StringIO

from blocks.extensions import Printing
from blocks.extensions.saveload import Checkpoint, SAVED_TO
from blocks.serialization import secure_dump


class StdoutLines(list):
    def __enter__(self):
        self._stringio = StringIO()
        self._stdout = sys.stdout
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class PrintingTo(Printing):
    def __init__(self, default, path, **kwargs):
        super(PrintingTo, self).__init__(**kwargs)
        self.path = path
        with open(self.path, "w") as f:
            f.truncate(0)
            f.writelines(default)
            f.write("\n")

    def do(self, *args, **kwargs):
        with StdoutLines() as lines:
            super(PrintingTo, self).do(*args, **kwargs)
        with open(self.path, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")


class PartsOnlyCheckpoint(Checkpoint):
    def __init__(self, path, **kwargs):
        super(PartsOnlyCheckpoint, self).__init__(path=path, **kwargs)
        self.iteration = 1

    def do(self, callback_name, *args):
        """Pickle the save_separately parts (and not the main loop object) to disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            ### this line is disabled from superclass impl to bypass using blocks.serialization.dump
            ### because pickling main thusly causes pickling error:
            ### "RuntimeError: maximum recursion depth exceeded while calling a Python object"
            # secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute), filenames[attribute] + '_%d.pkl' % self.iteration, pickle.dump,
                            protocol=pickle.HIGHEST_PROTOCOL)
            self.iteration += 1
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to + (path,))


class BestCheckpount(Checkpoint):
    def __init__(self, path, notification_name, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(BestCheckpount, self).__init__(path=path, **kwargs)
        self.notification_name = notification_name
        # self.iteration = 1

    def do(self, callback_name, *args):
        if self.notification_name in self.main_loop.log.current_row:
            _, from_user = self.parse_args(callback_name, args)
            try:
                path = self.path
                if from_user:
                    path, = from_user
                filenames = self.save_separately_filenames(path)
                for attribute in self.save_separately:
                    secure_dump(getattr(self.main_loop, attribute), filenames[attribute] + '.pkl', pickle.dump, protocol=pickle.HIGHEST_PROTOCOL)
                    # self.iteration += 1
            except Exception:
                path = None
                raise
            finally:
                already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
                self.main_loop.log.current_row[SAVED_TO] = (already_saved_to + (path,))
