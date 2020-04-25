import optparse
import sys
import threading
from pathlib import Path

from firebase import FirebaseHelper
from generic_builder import GenericBuilder
from saving_worker import SavingWorker


def spin_saving_worker(job_queue, firebase_helper, cv):
    saving_worker = SavingWorker(firebase_helper)
    with cv:
        cv.wait()
        if not len(job_queue) == 0:
            data = job_queue.pop(0)
            if data:
                saving_worker.save_output(data)


def spin_builder(finished_queue, firebase_helper, cv):
    # while True:
        jobs_enqeued = firebase_helper.get_jobs_enqeued()
        if jobs_enqeued:
            with cv:
                job = jobs_enqeued.pop(0)
                # firebase_helper.pop_job(job)

                if job:
                    job = firebase_helper.get_job_data(job)

                    job_files_path = Path(str(Path.home())+'/JobFiles/'+job.id)
                    job_files_path.mkdir(parents=True, exist_ok=False)

                    log_dir = str(job_files_path)+'/Tensorboard'
                    log_file = open(str(job_files_path)+'/log.txt', 'w+')
                    # sys.stderr = log_file
                    # sys.stdout = log_file
                    builder = GenericBuilder(job, log_dir, finished_queue, cv)
                    builder.build()
                    log_file.close()
                    # sys.stdout = sys.__stdout__
                    # sys.stderr = sys.__stderr__


if __name__ == "__main__":

    job_queue = []
    firebase_helper = FirebaseHelper()
    condition = threading.Condition()

    saver = threading.Thread(target=spin_saving_worker, args=(
        job_queue, firebase_helper, condition), daemon=True)

    # builder = threading.Thread(target=spin_builder, args=(job_queue, firebase_helper))

    saver.start()

    spin_builder(job_queue, firebase_helper, condition)
