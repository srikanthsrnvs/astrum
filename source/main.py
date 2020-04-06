import optparse
import sys

from generic_builder import GenericBuilder


def save_logs(path, tb_path):
    # TODO: save the logs after the script terminates by uploading the log to storage and saving in DB, also save the tensorboard logs


if __name__ == "__main__":

    log = open('log.out', 'w+')
    sys.stdout = log

    parser = optparse.OptionParser()

    parser.add_option('--urls',
                      action="store", dest="urls",
                      help="The location of the dataset", default="")
    parser.add_option('--type',
                      action="store", dest="type",
                      help="The type of network to build", default="")
    parser.add_option('--jobid',
                      action="store", dest="jobid",
                      help="The jobid of the job", default="")
    options, args = parser.parse_args()

    urls = options.urls.split(',')
    job_type = options.type
    job_id = options.job_id
    log_dir = 'logs/'
    builder = GenericBuilder(job_type, len(urls), urls, job_id, log_dir)
    builder.build()

    log.close()

    save_logs('log.out', log_dir)
