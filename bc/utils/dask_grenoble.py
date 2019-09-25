import os
import datetime

from dask_jobqueue import OARCluster


class AlpesCluster(OARCluster):
    def __init__(
            self,
            cores,
            name,
            processes=1,
            mem_req=4000,
            walltime='72:00:00',
            venv=None,
            to_source='~/.bashrc',
            log_dir='/home/apashevi/Logs/dask/',
            spill_dir='/home/apashevi/Logs/dask/',
            env_extra=None,
            besteffort=False,
            job_extra=None,
            interface_node=None,
            extra='',
            **kwargs):

        if name == 'dask-cpu':
            resource_spec = 'nodes=1/core={}'.format(cores)
        elif name == 'dask-gpu':
            resource_spec = None
        else:
            raise NotImplementedError
        name += '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        os.path.join(log_dir, 'logs')

        if besteffort:
            if job_extra is None:
                job_extra = []
            job_extra += [' -t besteffort -t idempotent']

        job_extra += [
            '--stdout={}'.format(os.path.join(log_dir, '%jobid%_stdout.txt'))
        ]
        job_extra += [
            '--stderr={}'.format(os.path.join(log_dir, '%jobid%_stderr.txt'))
        ]

        OARCluster.__init__(
            self,
            resource_spec=resource_spec,
            walltime=walltime,
            name=name,
            cores=cores,
            processes=processes,
            memory='{}m'.format(mem_req),
            local_directory=spill_dir,
            extra=extra,
            env_extra=env_extra,
            job_extra=job_extra,
            interface_node=interface_node,
            **kwargs)


class CPUCluster(AlpesCluster):
    def __init__(self, ncpus=1, **kwargs):
        cores = ncpus
        AlpesCluster.__init__(self, cores=cores, name='dask-cpu', **kwargs)


class GPUCluster(AlpesCluster):
    def __init__(self, **kwargs):
        job_extra = [
            '-p \'not host=\'\"\'\"\'gpuhost23\'\"\'\"\' and not host=\'\"\'\"\'gpuhost24\'\"\'\"\' and not host=\'\"\'\"\'gpuhost25\'\"\'\"\' and not host=\'\"\'\"\'gpuhost26\'\"\'\"\' and not host=\'\"\'\"\'gpuhost27\'\"\'\"\'\''
        ]
        AlpesCluster.__init__(
            self, cores=1, name='dask-gpu', job_extra=job_extra, **kwargs)
