import getpass
import os
from dask.distributed import Client, LocalCluster


def make_client(cluster, type_workers, num_workers, log_dir, no_nanny=False):
    """
    no_nanny option is there to allow workers to create their own workers.
    usefull if you have gpu workers creating their own cpu workers for data loading.
    """
    if no_nanny:
        extra = ['--no-nanny', '--no-bokeh']
        processes = False
    else:
        extra = []
        processes = True

    if cluster == 'paris':
        from dask_jobqueue import SGECluster

        job_extra = [
            '-pe serial 1', '--stdout={}'.format(
                os.path.join(log_dir, '%jobid%_stdout.txt')),
            '--stderr={}'.format(os.path.join(log_dir, '%jobid%_stderr.txt'))
        ]

        cluster = SGECluster(
            queue='gaia.q,chronos.q,titan.q,zeus.q',
            resource_spec='h_vmem=2000000M,mem_req=2000M',
            job_extra=['-pe serial 1'],
            env_extra=[
                'source /sequoia/data1/rstrudel/miniconda3/etc/profile.d/conda.sh',
                'conda activate bullet', 'export LANG=en_US.UTF-8',
                'export LC_ALL=en_US.UTF-8',
                'export PYTHONUNBUFFERED=non_empty'
            ],
            walltime='720:00:00',
            memory='4GB',
            extra=extra,
            cores=1,
            local_directory=os.path.join('/sequoia/data2', getpass.getuser(),
                                         'dask'))
        cluster.start_workers(num_workers)
    elif cluster == 'grenoble':
        from bc.utils.dask_grenoble import GPUCluster

        dask_log_dir = log_dir.replace('agents', 'dask').replace('/seed', '-s')
        if not os.path.exists(dask_log_dir):
            os.mkdir(dask_log_dir)
        cluster = GPUCluster(
            extra=['--no-nanny', '--no-bokeh'],
            walltime='72:00:00',
            log_dir=dask_log_dir,
            besteffort=True,
            interface_node='edgar',
        )
        # cluster.start_workers(num_gpus)
        cluster.adapt(minimum=0, maximum=num_workers)
    elif cluster == 'local':
        cluster = LocalCluster(processes=processes)
    else:
        raise ValueError('Unknown cluster name: {}'.format(cluster))

    client = Client(cluster)
    return client
