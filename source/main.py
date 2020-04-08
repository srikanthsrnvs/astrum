import optparse
import sys

from firebase import FirebaseHelper
from generic_builder import GenericBuilder

if __name__ == "__main__":

    while True:
        jobs_enqeued = FirebaseHelper().get_jobs_enqeued()
        if jobs_enqeued:
            job = jobs_enqeued.pop(0)
            FirebaseHelper().pop_job(job)

            if job:
                job_data, urls = FirebaseHelper().get_job_data(job)
                job_type = job_data['type']
                log_dir = job+'_logs/'
                log_file = open(job+'_output.txt', 'w+')
                sys.stderr = log_file
                sys.stdout = log_file
                builder = GenericBuilder(job_type, len(urls), job, log_dir, urls=urls)
                builder.build()
                log_file.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__



    # parser = optparse.OptionParser()

    # parser.add_option('--urls',
    #                   action="store", dest="urls",
    #                   help="The location of the dataset", default="")
    # parser.add_option('--type',
    #                   action="store", dest="type",
    #                   help="The type of network to build", default="")
    # parser.add_option('--job_id',
    #                   action="store", dest="job_id",
    #                   help="The job_id of the job", default="")
    # options, args = parser.parse_args()

    
