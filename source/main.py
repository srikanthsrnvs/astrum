import optparse
import sys

from firebase import FirebaseHelper
from generic_builder import GenericBuilder

if __name__ == "__main__":

    parser = optparse.OptionParser()

    parser.add_option('--urls',
                      action="store", dest="urls",
                      help="The location of the dataset", default="")
    parser.add_option('--type',
                      action="store", dest="type",
                      help="The type of network to build", default="")
    parser.add_option('--job_id',
                      action="store", dest="job_id",
                      help="The job_id of the job", default="")
    options, args = parser.parse_args()

    urls = options.urls.split(',')
    job_type = options.type
    job_id = options.job_id
    log_dir = job_id+'_logs/'
    builder = GenericBuilder(job_type, len(urls), job_id, log_dir, urls=urls)
    builder.build()
