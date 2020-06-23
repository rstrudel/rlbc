import os
import gym
import json
import click
import shutil
import torch
import pickle as pkl

from bc.dataset import DatasetWriter
from bc.dataset.utils import compress_images, process_trajectory
from bc.utils import videos, Report
from bc.utils.attention_hook import AttentionHook

from mime.agent import ScriptAgent, ReplayAgent
from bc.agent import BCAgent, RLAgent


class Collector:
    def __init__(self,
                 env,
                 agent,
                 collect_dir,
                 model_dir,
                 max_steps,
                 timescale,
                 skill_sequence,
                 record_failed=False,
                 db_type='demos',
                 report_path=None,
                 num_steps_buffer=100,
                 skill_collection=False,
                 render=False,
                 enforce_stop_when_done=False,
                 device='cpu',
                 image_augmentation='',
                 attention_maps=False,
                 replay_dir='',
                 replan_every=0,
                 **unused_kwargs):
        self.env_name = env
        self.agent_type = agent
        self.collect_dir = collect_dir
        self.replay_dir = replay_dir
        self.model_dir = model_dir
        self.dataset_type = db_type
        self.record_fails = record_failed
        self.num_steps_buffer = num_steps_buffer
        self.max_steps = max_steps
        self.skill_collection = skill_collection
        self.render = render
        self.enforce_stop_when_done = enforce_stop_when_done
        self.device = device
        self.attention_maps = attention_maps
        self.image_augmentation = image_augmentation

        assert self.agent_type in ('bc', 'rl', 'script', 'replay')
        assert self.dataset_type in ('demos', 'video', 'evaluation')

        if not os.path.isdir(self.collect_dir):
            os.mkdir(self.collect_dir)
        self.report_path = report_path

        # create the environment
        self._create_env()

        # skills related stuff
        self.timescales = timescale
        self.skill_sequence = skill_sequence
        if isinstance(self.timescales, int) and len(self.skill_sequence) > 0:
            timescales = {}
            for skill in range(max(self.skill_sequence) + 1):
                timescales[skill] = self.timescales
            self.timescales = timescales
        elif isinstance(self.timescales, list):
            timescales = {}
            for idx, timescale_skill in enumerate(timescale):
                timescales[idx] = timescale_skill
            self.timescales = timescales

    def __del__(self):
        self.env.close()

    def __call__(self, seed):
        success, seed, epoch, count = self.run_episode(seed)
        return success, seed, epoch, count

    def _create_env(self):
        self.env = gym.make(self.env_name)
        env_scene = self.env.unwrapped.scene
        if self.render:
            env_scene.renders(True)

    def _create_agent(self, agent_type, epoch, seed):
        if agent_type == 'script':
            agent = ScriptAgent(self.env)
        elif agent_type == 'bc':
            agent = BCAgent(self.model_dir, epoch, self.max_steps,
                            self.device, self.image_augmentation)
            agent.seed_exp(seed)
        elif agent_type == 'rl':
            agent = RLAgent(self.model_dir, epoch, self.max_steps,
                            self.device, self.env)
        elif agent_type == 'replay':
            agent = ReplayAgent(self.replay_dir)
            agent.set_seed(seed)

        # if agent is using rgb input
        # check that collection and agent evaluation are done both with or without egl
        # pybullet rgb rendering is not the same with and without egl
        if agent_type in ['bc', 'rl'] and 'rgb' in agent.model.args['input_type']:
            agent_env_name = agent.model.args['env_name']
            if 'EGL' in agent_env_name:
                assert 'EGL' in self.env_name
            else:
                assert 'EGL' not in self.env_name

        return agent

    def _init_episode(self, seed):
        if self.dataset_type in ['demos']:
            dataset_path = os.path.join(self.collect_dir, '{:06}'.format(seed))
            dataset_writer = DatasetWriter(
                dataset_path, self.env_name, rewrite=True)
            dataset_writer.init()
        else:
            dataset_writer = None
            dataset_path = None
        # plugin to change the environment after reset
        if self.skill_collection:
            self._activate_skill_collection(self.env)
        # write gripper max vel, lin max vel, ang max vel
        info_dataset = os.path.join(self.collect_dir, 'info.json')
        json.dump(dict(env_name=self.env_name), open(info_dataset, 'w'))
        # seed the environment
        self.env.seed(seed)
        return dataset_writer, dataset_path

    def _get_skill(self, count_master):
        if len(self.skill_sequence) > 0:
            assert count_master < len(self.skill_sequence)
            skill = self.skill_sequence[count_master]
        elif 'Cubes' in self.env_name and self.agent_type == 'bc':
            skill = self.env.unwrapped.scene.skill
        else:
            skill = None
        return skill

    def _get_action(self, obs, skill, agent):
        if self.agent_type == 'bc':
            action = agent.get_action(obs, skill)
        elif self.agent_type == 'rl':
            action = agent.get_action(obs)
        else:
            action = agent.get_action()
        return action

    def run_episode(self, seed_epoch):
        seed, epoch = seed_epoch
        observs, actions, steps, scalars = [], [], [], {}
        done = False
        count, count_master, count_skill = 0, 0, 0

        dataset_writer, dataset_path = self._init_episode(seed)
        obs = self.env.reset()
        agent = self._create_agent(self.agent_type, epoch, seed)

        # get an agent action and a predefined skill (if any)
        skill = self._get_skill(count_master)
        action = self._get_action(obs, skill, agent)
        info = {'success': False}

        # define attention hook
        attention_hook = None
        if self.attention_maps:
            assert self.agent_type in ['bc', 'rl']
            attention_hook = AttentionHook(agent.model.net.module)

        while action:
            # check if the skill should be skipped during data recording
            skill_is_dummy = self.agent_type == 'script' and 'skill' in obs and obs['skill'] == -1

            # prepare data for recording
            if self.dataset_type == 'demos':
                compress_images(obs)

            # store observations and actions if recording a dataset
            if self.dataset_type != 'evaluation':
                if not skill_is_dummy:
                    observs.append(obs)
                    actions.append(action)
                    steps.append(count)

            # do the environment step and update variables
            obs, reward, done, info = self.env.step(action)
            if not skill_is_dummy:
                count += 1
                count_skill += 1
                skill_is_done = (skill is not None and self.timescales is not None
                                 and count_skill == self.timescales[skill])
                if self.agent_type == 'bc' and skill_is_done:
                    count_skill = 0
                    count_master += 1

            # check whether the recording should be over
            if done:
                if len(info['failure_message']) > 0:
                    break
                if self.agent_type in ('bc', 'rl') or self.enforce_stop_when_done:
                    break
            if count_master == len(
                    self.skill_sequence) and len(self.skill_sequence) > 0:
                break

            # get an agent action and a predefined skill (if any)
            skill = self._get_skill(count_master)
            action = self._get_action(obs, skill, agent)
            # action = self._get_action(obs, None, agent)

            # blend attention maps
            if attention_hook is not None:
                assert 'rgb0' in obs
                if skill is not None:
                    attention_hook.skill = skill
                obs['rgb0'] = attention_hook.blend_map(obs['rgb0'])

            # if more than num_steps_buffer steps are performed,
            # update the images and scalars dataset with last observations, clean the buffer
            if self.dataset_type in ['demos'
                                     ] and len(observs) > 0 and count % self.num_steps_buffer == 0:
                frames_chunk, scalars_chunk = process_trajectory(
                    observs, actions, steps, seed)
                # write images into worker dataset
                dataset_writer.write_frames(frames_chunk)
                # update scalars
                scalars.update(scalars_chunk)
                observs, actions, steps = [], [], []

        # free up GPU memory
        del agent
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        self._write_data(
            dataset_writer=dataset_writer,
            dataset_path=dataset_path,
            success=info['success'],
            seed=seed,
            epoch=epoch,
            observs=observs,
            actions=actions,
            steps=steps,
            scalars=scalars)

        if not info['success']:
            if action is None:
                info['failure_message'] = 'Reached max steps.'
            # click.secho(
            #     'Failure Seed {}: {}'.format(seed, info['failure_message']),
            #     fg='red')

        return info['success'], seed, epoch, count

    def _activate_skill_collection(self, env):
        # plugin for collecting skills in BowlEnv
        env.unwrapped.scene.skill_data_collection = True

    def _write_data(self, dataset_writer, dataset_path, success, seed, epoch, observs,
                    actions, steps, scalars):
        # write residual chunk to dataset if success
        # remove dataset if filtering failures
        if self.dataset_type in ['demos']:
            if success or self.record_fails:
                # print('Writing trajectory {}'.format(seed))
                # update images and scalars dataset with last observations
                if observs:
                    frames_chunk, scalars_chunk = process_trajectory(
                        observs, actions, steps, seed)
                    # write images into worker dataset
                    dataset_writer.write_frames(frames_chunk)
                    # update scalars
                    scalars.update(scalars_chunk)
                # write scalars into a pickle file
                pkl.dump(scalars,
                         open(os.path.join(self.collect_dir,
                                           '{:06}.pkl'.format(seed)), 'wb'))
                dataset_writer.close()
            else:
                dataset_writer.close()
                # do not create pickle file for scalars
                # delete worker images dataset
                shutil.rmtree(dataset_path)
        elif self.dataset_type == 'video':
            skip = 1
            path_video = os.path.join(self.collect_dir, '{:06}.mp4'.format(seed))
            videos.write_video([obs['rgb0'] for obs in observs[::skip]], path_video)
        elif self.dataset_type == 'evaluation':
            Report.add_entry_asynch(self.report_path, seed, epoch, success)

