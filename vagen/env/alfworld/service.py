import os
import time
import logging
import threading
import multiprocessing as mp
from queue import Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from vagen.env.base.base_service import BaseService
from vagen.env.alfworld.env import ALFWorldEnv
from vagen.env.alfworld.env_config import ALFWorldEnvConfig
from vagen.env.alfworld.service_config import ALFWorldServiceConfig
from vagen.server.serial import serialize_observation, serialize_step_result, serialize_info

class ALFWorldService(BaseService):
    """
    Service class for ALFWorld environments using multiprocessing and multithreading.
    Provides batch operations: create, reset, step, compute_reward, get_system_prompts, close.
    """
    def __init__(self, config: ALFWorldServiceConfig):
        # Determine CPU count
        cpu_count = os.cpu_count() or 4
        # Max processes from config.max_workers
        self.max_process_workers = min(config.get('max_workers', cpu_count), cpu_count)
        # Threads per process
        self.max_thread_workers = config.max_thread_workers
        # Timeout for inter-process commands (seconds)
        self.timeout = config.timeout

        # Mapping: env_id -> process_id
        self.environments = {}
        # Store raw configs
        self.env_configs = {}
        # Worker processes list
        self.processes = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ALFWorldService')

        # Setup MP queues
        self._setup_mp_queues()

    def _setup_mp_queues(self):
        self.task_queues = []
        self.result_queues = []
        for _ in range(self.max_process_workers):
            self.task_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())

    def _worker_process(self, process_id, task_queue, result_queue, max_thread_workers):
        """
        Target function run in each subprocess to manage ALFWorldEnv instances.
        """
        local_environments = {}
        local_env_configs = {}
        # Thread pool for potential intra-process parallelism
        thread_pool = ThreadPoolExecutor(max_workers=max_thread_workers)
        running = True
        
        #@ TODO revise this
        def send_heartbeat():
            while running:
                try:
                    result_queue.put((-999, "heartbeat", process_id))
                    time.sleep(120)
                except:
                    pass
        
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()
    
        while running:
            try:
                command, task_id, args = task_queue.get()
                if command == "create":
                    env_id, config = args 
                    try:
                        env_name = config.get('env_name', 'alfworld')
                        if env_name != 'alfworld':
                            result_queue.put((task_id, "error", f"Expected environment type 'alfworld', got '{env_name}'"))
                            continue
                        
                        env_config_dict = config.get('env_config', {})
                        env_config = ALFWorldEnvConfig(**env_config_dict)
                        
                        env = ALFWorldEnv(env_config)
                        
                        local_environments[env_id] = env
                        local_env_configs[env_id] = env_config
                        
                        result_queue.put((task_id, "success", env_id))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error creating environment {env_id}: {str(e)}"))

                elif command == 'reset':
                    env_id, seed = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        obs, info = local_environments[env_id].reset(seed)
                        s_info = serialize_info(info)
                        s_obs = serialize_observation(obs)
                        result_queue.put((task_id, 'success', (s_obs, s_info)))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))

                elif command == 'step':
                    env_id, action = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        obs, reward, done, info = local_environments[env_id].step(action)
                        serialized = serialize_step_result((obs, reward, done, info))
                        result_queue.put((task_id, 'success', serialized))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))

                elif command == 'compute_reward':
                    env_id = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        r = local_environments[env_id].compute_reward()
                        result_queue.put((task_id, 'success', r))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))

                elif command == 'system_prompt':
                    env_id = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        p = local_environments[env_id].system_prompt()
                        result_queue.put((task_id, 'success', p))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))

                elif command == 'close':
                    env_id = args
                    if env_id in local_environments:
                        try:
                            local_environments[env_id].close()
                        except:
                            pass
                        local_environments.pop(env_id, None)
                        local_env_configs.pop(env_id, None)
                    result_queue.put((task_id, 'success', True))

                elif command == 'exit':
                    running = False
                    result_queue.put((task_id, 'success', 'exited'))

                else:
                    result_queue.put((task_id, 'error', f"Unknown command {command}"))

            except Exception as e:
                try:
                    result_queue.put((-1, 'error', str(e)))
                except:
                    pass
        # Cleanup
        thread_pool.shutdown()

    def _start_worker_processes(self):
        for i in range(self.max_process_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(i, self.task_queues[i], self.result_queues[i], self.max_thread_workers),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            self.logger.info(f"Started ALFWorld worker {i} (pid={p.pid})")

    def _assign_to_process(self, env_id):
        loads = [0] * self.max_process_workers
        for pid in self.environments.values():
            loads[pid] += 1
        return loads.index(min(loads))

    def _send_command(self, pid, command, env_id, args):
        task_id = hash(f"{command}_{env_id}_{time.time()}")
        self.task_queues[pid].put((command, task_id, args))
        while True:
            try:
                rid, status, result = self.result_queues[pid].get(timeout=self.timeout)
                
                #@ TODO revise this
                if rid == -999 and status == "heartbeat":
                    last_heartbeat = time.time()
                    self.logger.debug(f"Received heartbeat from worker {pid}")
                    continue
            
                if rid != task_id:
                    continue
                if status == 'success':
                    return result
                raise Exception(result)
            except Empty:
                raise TimeoutError(f"Timeout {command} for {env_id}")

    def create_environments_batch(self, ids2configs):        
        if not self.processes:
            self._start_worker_processes()
        
        # Group environments by assigned process
        by_pid = {}
        for env_id, cfg in ids2configs.items():
            pid = self._assign_to_process(env_id)
            by_pid.setdefault(pid, []).append((env_id, cfg))
            # Store assigned process ID and config for later use
            self.environments[env_id] = pid
            self.env_configs[env_id] = cfg
        
        # Define function to create environments within a process group
        def _create_group(pid, group):
            results = {}
            
            # Define single environment creation function
            def _create_single_env(env_id, cfg):
                try:
                    result = self._send_command(pid, 'create', env_id, (env_id, cfg))
                    return env_id, result, None
                except Exception as e:
                    error_msg = f"Error creating environment {env_id}: {str(e)}"
                    self.logger.error(error_msg)
                    return env_id, None, error_msg
            
            # Process environments sequentially within each process group
            for env_id, cfg in group:
                env_id, result, error = _create_single_env(env_id, cfg)
                if error:
                    # Using dict.pop with a default (None) to prevent KeyError
                    self.environments.pop(env_id, None)
                    self.env_configs.pop(env_id, None)
                    results[env_id] = f"Error: {error}"
                else:
                    results[env_id] = result
                    
            return results
        
        # Use ThreadPoolExecutor to parallelize across processes
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            futures = []
            # Submit creation tasks for each process group
            for pid, group in by_pid.items():
                futures.append(exe.submit(_create_group, pid, group))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()

    def reset_batch(self, ids2seeds):
        results = {}
        by_pid = {}
        for env_id, seed in ids2seeds.items():
            pid = self.environments.get(env_id)
            by_pid.setdefault(pid, []).append((env_id, seed))
        def _reset_group(pid, group):
            out = {}
            for env_id, seed in group:
                obs, info = self._send_command(pid, 'reset', env_id, (env_id, seed))
                out[env_id] = (obs, info)
            return out
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            for pid, group in by_pid.items():
                futures = exe.submit(_reset_group, pid, group)
                results.update(futures.result())
        return results

    def step_batch(self, ids2actions):        
        results = {}
        by_pid = {}
        
        # Group environments by process ID
        for env_id, action in ids2actions.items():
            pid = self.environments.get(env_id)
            by_pid.setdefault(pid, []).append((env_id, action))
        
        # Define function to handle stepping environments in each process
        def _step_group(pid, group):
            out = {}
            
            # Use ThreadPoolExecutor for parallel steps within each process group
            with ThreadPoolExecutor(max_workers=min(len(group), self.max_thread_workers)) as local_exe:
                step_futures = {}
                
                # Function to step a single environment
                def _step_single_env(env_id, action):
                    try:
                        result = self._send_command(pid, 'step', env_id, (env_id, action))
                        return env_id, result, None
                    except Exception as e:
                        error_msg = f"Error stepping environment {env_id}: {str(e)}"
                        self.logger.error(error_msg)
                        return env_id, None, error_msg
                
                # Submit all environments in this group
                for env_id, action in group:
                    step_futures[local_exe.submit(_step_single_env, env_id, action)] = env_id
                
                # Process results as they complete
                for future in as_completed(step_futures):
                    env_id, result, error = future.result()
                    if error:
                        out[env_id] = ({}, 0.0, True, {"error": error})
                    else:
                        out[env_id] = result
            
            return out
        
        # Use ThreadPoolExecutor to parallelize across processes
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            futures = []
            
            # Submit stepping tasks for each process group
            for pid, group in by_pid.items():
                futures.append(exe.submit(_step_group, pid, group))
            
            # Process results as they complete
            for future in as_completed(futures):
                results.update(future.result())
        
        return results

    def compute_reward_batch(self, env_ids):
        rewards = {}
        by_pid = {}
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            by_pid.setdefault(pid, []).append(env_id)
        def _reward_group(pid, group):
            out = {}
            for env_id in group:
                out[env_id] = self._send_command(pid, 'compute_reward', env_id, env_id)
            return out
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            for pid, group in by_pid.items():
                futures = exe.submit(_reward_group, pid, group)
                rewards.update(futures.result())
        return rewards

    def get_system_prompts_batch(self, env_ids):
        prompts = {}
        by_pid = {}
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            by_pid.setdefault(pid, []).append(env_id)
        def _prompt_group(pid, group):
            out = {}
            for env_id in group:
                out[env_id] = self._send_command(pid, 'system_prompt', env_id, env_id)
            return out
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            for pid, group in by_pid.items():
                futures = exe.submit(_prompt_group, pid, group)
                prompts.update(futures.result())
        return prompts

    def close_batch(self, env_ids=None):
        if env_ids is None:
            env_ids = list(self.environments.keys())
        by_pid = {}
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            by_pid.setdefault(pid, []).append(env_id)
        def _close_group(pid, group):
            for env_id in group:
                self._send_command(pid, 'close', env_id, env_id)
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as exe:
            for pid, group in by_pid.items():
                exe.submit(_close_group, pid, group)
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    def __del__(self):
        try:
            self.close_batch()
        except:
            pass
        for i, q in enumerate(self.task_queues):
            try:
                q.put(('exit', -1, None))
            except:
                pass
        for p in self.processes:
            try:
                p.join(timeout=self.timeout)
                if p.is_alive():
                    p.terminate()
            except:
                pass