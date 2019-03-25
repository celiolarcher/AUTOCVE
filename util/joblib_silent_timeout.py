from joblib import Parallel
from multiprocessing import TimeoutError
from joblib.my_exceptions import TransportableException
from joblib.externals.loky.process_executor import TerminatedWorkerError

import time

class ParallelSilentTimeout(Parallel):
    def __init__(self, *args, **kwargs):
        if 'batch_size' in kwargs:
             print("Warning: batch_size is set to 1 in ParallelSilentTimeout, since it's required to its correct operation.")
        kwargs['batch_size']=1
        super(ParallelSilentTimeout, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        backend = self._backend
        managed_backend = self._managed_backend

        if (backend is not None and hasattr(backend, 'abort_everything')):
             try:
                 self._managed_backend=False
                 output = super(ParallelSilentTimeout, self).__call__(*args, **kwargs)
             finally:
                 if managed_backend:
                      self._backend.abort_everything(ensure_ready=False) #abort_everything allows to shutdown the workers in the loky backend
                 self._managed_backend = managed_backend
        else:
             output = super(ParallelSilentTimeout, self).__call__(*args, **kwargs)

        return output

    def retrieve(self):
        self._output = list()
        while self._iterating or len(self._jobs) > 0:
            if len(self._jobs) == 0:
                # Wait for an async callback to dispatch new jobs
                time.sleep(0.01)
                continue
            # We need to be careful: the job list can be filling up as
            # we empty it and Python list are not thread-safe by default hence
            # the use of the lock
            with self._lock:
                job = self._jobs.pop(0)

            try:
                start_time = time.time()
                if getattr(self._backend, 'supports_timeout', False):
                    self._output.extend(job.get(timeout=self.timeout))
                else:
                    self._output.extend(job.get())

            except TimeoutError as exception:
                # batch_size fixed as 1, since the TimeoutError is attributed just for the pipeline which reach the timeout (in batch_size bigger than one, all pipelines in a batch receive the TimeoutError value)
                cb_batch_size = 1
                self._output.extend(cb_batch_size * [TimeoutError()])
                #self._output.extend([None])
                # need to dispatch potential next task because
                # batch completion callback never got executed
                if self._original_iterator is not None:
                    self.dispatch_next()

            except TerminatedWorkerError as exception:
                # batch_size fixed as 1, since the TimeoutError is attributed just for the pipeline which reach the timeout (in batch_size bigger than one, all pipelines in a batch receive the TimeoutError value)
                cb_batch_size = 1
                self._output.extend(cb_batch_size * [TerminatedWorkerError()])
                #self._output.extend([None])
                # need to dispatch potential next task because
                # batch completion callback never got executed
                if self._original_iterator is not None:
                    self.dispatch_next()

            except BaseException as exception:
                # Note: we catch any BaseException instead of just Exception
                # instances to also include KeyboardInterrupt.

                # Stop dispatching any new job in the async callback thread
                self._aborting = True

                # If the backend allows it, cancel or kill remaining running
                # tasks without waiting for the results as we will raise
                # the exception we got back to the caller instead of returning
                # any result.
                backend = self._backend
                if (backend is not None and
                        hasattr(backend, 'abort_everything')):
                    # If the backend is managed externally we need to make sure
                    # to leave it in a working state to allow for future jobs
                    # scheduling.
                    ensure_ready = self._managed_backend
                    backend.abort_everything(ensure_ready=ensure_ready)

                if isinstance(exception, TransportableException):
                    # Capture exception to add information on the local
                    # stack in addition to the distant stack
                    this_report = format_outer_frames(context=10,
                                                      stack_start=1)
                    raise exception.unwrap(this_report)
                else:
                    raise




