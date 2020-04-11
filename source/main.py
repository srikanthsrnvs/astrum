import optparse
import sys

from firebase import FirebaseHelper
from generic_builder import GenericBuilder

if __name__ == "__main__":

    

    # while True:
        jobs_enqeued = FirebaseHelper().get_jobs_enqeued()
        if jobs_enqeued:
            job = jobs_enqeued.pop(0)
            # FirebaseHelper().pop_job(job)

            if job:
                job = FirebaseHelper().get_job_data(job)
                
                log_dir = job.id+'_logs/'
                log_file = open(job.id+'_output.txt', 'w+')
                # sys.stderr = log_file
                # sys.stdout = log_file
                builder = GenericBuilder(job, log_dir)
                builder.build()
                log_file.close()
                # sys.stdout = sys.__stdout__
                # sys.stderr = sys.__stderr__
