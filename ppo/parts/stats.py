from collections import deque
import numpy as np


def init(num_processes, eval=False):
    if not eval:
        # in the beginning of the training we want to have 100 values as well
        return_deque = deque([0]*100, maxlen=100)
        success_deque = deque([0]*100, maxlen=100)
    else:
        return_deque = deque(maxlen=100)
        success_deque = deque(maxlen=100)
    stats_global = {'return': return_deque,
                    'length': deque(maxlen=100),
                    'fail': deque(maxlen=100),
                    'fail_joints': deque(maxlen=100),
                    'fail_workspace': deque(maxlen=100),
                    'fail_objects': deque(maxlen=100),
                    'fail_crash': deque(maxlen=100),
                    'success': success_deque}
    stats_local = {'return': np.array([0] * num_processes, dtype=np.float32),
                   'done_before': np.array([False] * num_processes, dtype=np.bool)}
    return stats_global, stats_local


def update(stats_g, stats_l, reward, done, infos, args, overwrite_terminated=True):
    stats_l['return'] += reward[:, 0].cpu().numpy()
    if not overwrite_terminated:
        # for evaluation we want to run N envs and wait the N results
        # we do not want a short episode (e.g. two fails) to replace a longer one (e.g. a success)
        done_new = np.zeros(done.shape, dtype=np.bool)
        for idx, (done_now, done_before) in enumerate(zip(done, stats_l['done_before'])):
            if done_now and not done_before:
                done_new[idx] = True
        stats_l['done_before'] = np.logical_or(done, stats_l['done_before'])
        done = done_new

    # append stats of the envs that are done (reset or fail)
    stats_g['return'].extend(stats_l['return'][np.where(done)])
    infos_done = np.array(infos)[np.where(done)]
    stats_g['length'].extend([info['length'] for info in infos_done])
    success_done = [int(info['success']) for info in infos_done]
    stats_g['success'].extend(success_done)
    fail_messages_done = [info['failure_message'] for info in infos_done]
    num_done = int(np.sum(done))
    num_fail = int(np.sum([len(m) > 0 for m in fail_messages_done]))
    num_fail_joints = int(np.sum(['Joint' in m for m in fail_messages_done]))
    num_fail_workspace = int(np.sum(['Workspace' in m for m in fail_messages_done]))
    num_fail_objects = int(np.sum(['All the objects' in m for m in fail_messages_done]))
    num_fail_crash = int(np.sum(['Env crashed' in m for m in fail_messages_done]))
    stats_g['fail'].extend([1] * num_fail + [0] * (num_done - num_fail))
    stats_g['fail_joints'].extend([1] * num_fail_joints + [0] * (num_done - num_fail_joints))
    stats_g['fail_workspace'].extend([1] * num_fail_workspace + [0] * (num_done - num_fail_workspace))
    stats_g['fail_objects'].extend([1] * num_fail_objects + [0] * (num_done - num_fail_objects))
    stats_g['fail_crash'].extend([1] * num_fail_crash + [0] * (num_done - num_fail_crash))
    # zero out returns of the envs that are done (reset or fail)
    stats_l['return'][np.where(done)] = 0
    return stats_g, stats_l
