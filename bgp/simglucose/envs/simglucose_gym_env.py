from bgp.simglucose.simulation.env import T1DSimEnv
from bgp.simglucose.patient.t1dpatient import T1DPatientNew
from bgp.simglucose.sensor.cgm import CGMSensor
from bgp.simglucose.actuator.pump import InsulinPump
from bgp.simglucose.simulation.scenario_gen import (
    RandomBalancedScenario,
    SemiRandomBalancedScenario,
    CustomBalancedScenario,
)
from bgp.simglucose.controller.base import Action
from bgp.simglucose.analysis.risk import magni_risk_index
from bgp.rl import reward_functions
from bgp.rl.helpers import Seed
from bgp.rl import pid
import bgp.simglucose.controller.basal_bolus_ctrller as bbc

from importlib import resources

import pandas as pd
import numpy as np
import joblib
import copy
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import warnings
import logging

from copy import deepcopy
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import make_multi_agent

warnings.simplefilter(action="ignore", category=FutureWarning)


def reward_name_to_function(reward_name):
    if reward_name == "risk_diff":
        reward_fun = reward_functions.risk_diff
    elif reward_name == "risk_diff_bg":
        reward_fun = reward_functions.risk_diff_bg
    elif reward_name == "risk":
        reward_fun = reward_functions.reward_risk
    elif reward_name == "risk_bg":
        reward_fun = reward_functions.risk_bg
    elif reward_name == "risk_high_bg":
        reward_fun = reward_functions.risk_high_bg
    elif reward_name == "risk_low_bg":
        reward_fun = reward_functions.risk_low_bg
    elif reward_name == "magni_bg":
        reward_fun = reward_functions.magni_reward
    elif reward_name == "magni_misweight":
        reward_fun = reward_functions.magni_misweight
    elif reward_name == "cameron_bg":
        reward_fun = reward_functions.cameron_reward
    elif reward_name == "eps_risk":
        reward_fun = reward_functions.epsilon_risk
    elif reward_name == "target_bg":
        reward_fun = reward_functions.reward_target
    elif reward_name == "cgm_high":
        reward_fun = reward_functions.reward_cgm_high
    elif reward_name == "bg_high":
        reward_fun = reward_functions.reward_bg_high
    elif reward_name == "cgm_low":
        reward_fun = reward_functions.reward_cgm_low
    elif reward_name == "risk_insulin":
        reward_fun = reward_functions.risk_insulin
    elif reward_name == "magni_bg_insulin":
        reward_fun = reward_functions.magni_bg_insulin
    elif reward_name == "magni_bg_insulin_true":
        reward_fun = reward_functions.magni_bg_insulin_true
    elif reward_name == "threshold_bg":
        reward_fun = reward_functions.threshold
    elif reward_name == "expected_patient_cost":
        reward_fun = reward_functions.expected_patient_cost
    else:
        raise ValueError("{} not a proper reward_name".format(reward_name))
    return reward_fun


logger = logging.getLogger(__name__)


class SimglucoseEnv(gym.Env):
    """
    A gym environment supporting SAC learning. Uses PID control for initialization
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config={}, **kwargs):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        config.update(kwargs)
        self.source_dir = config["source_dir"]
        with resources.path("bgp.simglucose", "__init__.py") as data_path:
            data_path = data_path.parent
            self.patient_para_file = data_path / "params" / "vpatient_params.csv"
            self.control_quest = data_path / "params" / "Quest2.csv"
            self.pid_para_file = data_path / "params" / "pid_params.csv"
            self.pid_env_path = data_path / "params"
            self.sensor_para_file = data_path / "params" / "sensor_params.csv"
            self.insulin_pump_para_file = data_path / "params" / "pump_params.csv"
        # reserving half of pop for testing
        self.universe = (
            ["child#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
            + ["adolescent#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
            + ["adult#0{}".format(str(i).zfill(2)) for i in range(1, 6)]
        )
        self.universal = config["universal"]
        if config["seeds"] is None:
            seed_list = self._seed()
            config["seeds"] = Seed(
                numpy_seed=seed_list[0],
                sensor_seed=seed_list[1],
                scenario_seed=seed_list[2],
            )
        if config["patient_name"] is None:
            if self.universal:
                config["patient_name"] = np.random.choice(self.universe)
            else:
                config["patient_name"] = "adolescent#001"
        np.random.seed(config["seeds"]["numpy"])
        self.horizon = config["horizon"]
        self.seeds = config["seeds"]
        self.sample_time = 5
        self.day = int(1440 / self.sample_time)
        self.state_hist = int((config["n_hours"] * 60) / self.sample_time)
        self.time = config["time"]
        self.meal = config["meal"]
        self.norm = config["norm"]
        self.gt = config["gt"]
        self.reward_bias = config["reward_bias"]
        self.proxy_reward_fn = reward_name_to_function(config["proxy_reward_fun"])
        self.true_reward_fn = reward_name_to_function(config["true_reward_fun"])
        if config["reward_fun"] == "proxy":
            self.reward_fun = self.proxy_reward_fn
        else:
            self.reward_fun = self.true_reward_fn
        self.true_rew = 0
        self.rew = 0
        self.action_cap = config["action_cap"]
        self.action_bias = config["action_bias"]
        self.action_scale = config["action_scale"]
        self.basal_scaling = config["basal_scaling"]
        self.meal_announce = config["meal_announce"]
        self.meal_duration = config["meal_duration"]
        self.deterministic_meal_size = config["deterministic_meal_size"]
        self.deterministic_meal_time = config["deterministic_meal_time"]
        self.deterministic_meal_occurrence = config["deterministic_meal_occurrence"]
        self.residual_basal = config["residual_basal"]
        self.residual_bolus = config["residual_bolus"]
        self.carb_miss_prob = config["carb_miss_prob"]
        self.carb_error_std = config["carb_error_std"]
        self.residual_PID = config["residual_PID"]
        self.use_pid_load = config["use_pid_load"]
        self.fake_gt = config["fake_gt"]
        self.fake_real = config["fake_real"]
        self.suppress_carbs = config["suppress_carbs"]
        self.limited_gt = config["limited_gt"]
        self.termination_penalty = config["termination_penalty"]
        self.target = 140
        self.low_lim = 140  # Matching BB controller
        self.cooldown = 180
        self.last_cf = self.cooldown + 1
        self.start_date = config["start_date"]
        self.rolling_insulin_lim = config["rolling_insulin_lim"]
        self.rolling = []
        if self.start_date is None:
            start_time = datetime(2018, 1, 1, 0, 0, 0)
        else:
            start_time = datetime(
                self.start_date.year,
                self.start_date.month,
                self.start_date.day,
                0,
                0,
                0,
            )
        self.use_only_during_day = config["use_only_during_day"]

        if self.use_only_during_day:
            start_time = datetime(2018, 1, 1, 5, 0, 0)
        assert config["bw_meals"]  # otherwise code wouldn't make sense
        if config["reset_lim"] is None:
            self.reset_lim = {"lower_lim": 10, "upper_lim": 1000}
        else:
            self.reset_lim = config["reset_lim"]
        self.load = config["load"]
        self.hist_init = config["hist_init"]
        self.env = None
        self.use_old_patient_env = config["use_old_patient_env"]
        self.model = config["model"]
        self.model_device = config["model_device"]
        self.use_model = config["use_model"]
        self.harrison_benedict = config["harrison_benedict"]
        self.restricted_carb = config["restricted_carb"]
        self.unrealistic = config["unrealistic"]
        self.start_time = start_time
        self.time_std = config["time_std"]
        self.weekly = config["weekly"]
        self.update_seed_on_reset = config["update_seed_on_reset"]
        self.use_custom_meal = config["use_custom_meal"]
        self.custom_meal_num = config["custom_meal_num"]
        self.custom_meal_size = config["custom_meal_size"]
        self.patient_name = config["patient_name"]
        self.reward_scale: float = config.get("reward_scale", 1)
        self.is_baseline = config.get("is_baseline", False)
        self.set_patient_dependent_values(
            self.patient_name, noise_scale=config["noise_scale"]
        )
        self.env.scenario.day = 0

        # Baseline
        self.cnt = bbc.ManualBBController(
            target=self.target,
            cr=self.CR,
            cf=self.CF,
            basal=self.ideal_basal,
            sample_rate=self.sample_time,
            use_cf=True,
            use_bol=True,
            cooldown=self.cooldown,
            corrected=True,
            use_low_lim=True,
            low_lim=self.low_lim,
        )

    def get_true_rew(self):
        return self.true_rew

    def get_obs_rew(self):
        return self.rew

    def pid_load(self, n_days):
        for i in range(n_days * self.day):
            b_val = self.pid.step(self.env.CGM_hist[-1])
            act = Action(basal=0, bolus=b_val)
            _ = self.env.step(action=act, reward_fun=self.reward_fun, cho=None)

    def step(self, action):
        self.t += 1
        obs, reward, done, info = self._step(action, cho=None)
        reward *= self.reward_scale
        return obs, reward, done, info

    def translate(self, action):
        if self.action_scale == "basal":
            # 288 samples per day, bolus insulin should be 75% of insulin dose
            # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
            # https://care.diabetesjournals.org/content/34/5/1089
            action = (action + self.action_bias) * (
                (self.ideal_basal * self.basal_scaling) / (1 + self.action_bias)
            )
        else:
            action = (action + self.action_bias) * self.action_scale
        return max(0, action)

    def _step(self, action, cho=None, use_action_scale=True):
        # cho controls if carbs are eaten, else taken from meal policy
        if type(action) is np.ndarray:
            action = action.item()
        ma = self.announce_meal(5)
        carbs = ma[0]
        if np.random.uniform() < self.carb_miss_prob:
            carbs = 0
        error = np.random.normal(0, self.carb_error_std)
        carbs = carbs + carbs * error
        glucose = self.env.CGM_hist[-1]
        if use_action_scale:
            if self.action_scale == "basal":
                # 288 samples per day, bolus insulin should be 75% of insulin dose
                # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
                # https://care.diabetesjournals.org/content/34/5/1089
                action = (action + self.action_bias) * (
                    (self.ideal_basal * self.basal_scaling) / (1 + self.action_bias)
                )
            else:
                action = (action + self.action_bias) * self.action_scale
        if self.residual_basal:
            action += self.ideal_basal
        if self.residual_bolus:
            if carbs > 0:
                carb_correct = carbs / self.CR
                hyper_correct = (
                    (glucose > self.target) * (glucose - self.target) / self.CF
                )
                hypo_correct = (
                    (glucose < self.low_lim) * (self.low_lim - glucose) / self.CF
                )
                bolus = 0
                if self.last_cf > self.cooldown:
                    bolus += hyper_correct - hypo_correct
                bolus += carb_correct
                action += bolus / 5.0
                self.last_cf = 0
            self.last_cf += 5
        if self.residual_PID:
            action += self.pid.step(self.env.CGM_hist[-1])
        if self.action_cap is not None:
            action = min(self.action_cap, action)
        if self.rolling_insulin_lim is not None:
            if np.sum(self.rolling + [action]) > self.rolling_insulin_lim:
                action = max(
                    0,
                    action
                    - (np.sum(self.rolling + [action]) - self.rolling_insulin_lim),
                )
            self.rolling.append(action)
            if len(self.rolling) > 12:
                self.rolling = self.rolling[1:]
        if self.is_baseline:
            act = self.cnt.manual_bb_policy(carbs=carbs, glucose=glucose)
        else:
            act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(
            action=act,
            reward_fun=self.reward_fun,
            cho=cho,
            true_reward_fn=self.true_reward_fn,
            proxy_reward_fn=self.proxy_reward_fn,
        )
        info["glucose_controller_actions"] = act.basal + act.bolus
        info["glucose_pid_controller"] = self.pid.step(self.env.CGM_hist[-1])
        state = self.get_state(self.norm)
        done = self.is_done()
        if done and self.t < self.horizon and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        reward = reward + self.reward_bias
        if self.use_only_during_day and (
            self.env.time.hour > 20 or self.env.time.hour < 5
        ):
            done = True
        info["reward"] = reward
        return state, reward, done, info

    def announce_meal(self, meal_announce=None):
        t = (
            self.env.time.hour * 60 + self.env.time.minute
        )  # Assuming 5 minute sampling rate
        for i, m_t in enumerate(self.env.scenario.scenario["meal"]["time"]):
            # round up to nearest 5
            if m_t % 5 != 0:
                m_tr = m_t - (m_t % 5) + 5
            else:
                m_tr = m_t
            if meal_announce is None:
                ma = self.meal_announce
            else:
                ma = meal_announce
            if t < m_tr <= t + ma:
                return self.env.scenario.scenario["meal"]["amount"][i], m_tr - t
        return 0, 0

    def calculate_iob(self):
        ins = self.env.insulin_hist
        return np.dot(np.flip(self.iob, axis=0)[-len(ins) :], ins[-len(self.iob) :])

    def get_state(self, normalize=False):
        bg = self.env.CGM_hist[-self.state_hist :]
        insulin = self.env.insulin_hist[-self.state_hist :]
        if normalize:
            bg = np.array(bg) / 400.0
            insulin = np.array(insulin) * 10
        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate(
                (np.full(self.state_hist - len(insulin), -1), insulin)
            )
        return_arr = [bg, insulin]
        if self.time:
            time_dt = self.env.time_hist[-self.state_hist :]
            time = np.array(
                [(t.minute + 60 * t.hour) / self.sample_time for t in time_dt]
            )
            sin_time = np.sin(time * 2 * np.pi / self.day)
            cos_time = np.cos(time * 2 * np.pi / self.day)
            if normalize:
                pass  # already normalized
            if len(sin_time) < self.state_hist:
                sin_time = np.concatenate(
                    (np.full(self.state_hist - len(sin_time), -1), sin_time)
                )
            if len(cos_time) < self.state_hist:
                cos_time = np.concatenate(
                    (np.full(self.state_hist - len(cos_time), -1), cos_time)
                )
            return_arr.append(sin_time)
            return_arr.append(cos_time)
            if self.weekly:
                # binary flag signalling weekend
                if self.env.scenario.day == 5 or self.env.scenario.day == 6:
                    return_arr.append(np.full(self.state_hist, 1))
                else:
                    return_arr.append(np.full(self.state_hist, 0))
        if self.meal:
            cho = self.env.CHO_hist[-self.state_hist :]
            if normalize:
                cho = np.array(cho) / 20.0
            if len(cho) < self.state_hist:
                cho = np.concatenate((np.full(self.state_hist - len(cho), -1), cho))
            return_arr.append(cho)
        if self.meal_announce is not None:
            meal_val, meal_time = self.announce_meal()
            future_cho = np.full(self.state_hist, meal_val)
            return_arr.append(future_cho)
            future_time = np.full(self.state_hist, meal_time)
            return_arr.append(future_time)
        if self.fake_real:
            state = self.env.patient.state
            return np.stack([state for _ in range(self.state_hist)]).T.flatten()
        if self.gt:
            if self.fake_gt:
                iob = self.calculate_iob()
                cgm = self.env.CGM_hist[-1]
                if normalize:
                    state = np.array([cgm / 400.0, iob * 10])
                else:
                    state = np.array([cgm, iob])
            else:
                state = self.env.patient.state
            if self.meal_announce is not None:
                meal_val, meal_time = self.announce_meal()
                state = np.concatenate((state, np.array([meal_val, meal_time])))
            if normalize:
                # just the average of 2 days of adult#001, these values are patient-specific
                norm_arr = np.array(
                    [
                        4.86688301e03,
                        4.95825609e03,
                        2.52219425e03,
                        2.73376341e02,
                        1.56207049e02,
                        9.72051746e00,
                        7.65293763e01,
                        1.76808549e02,
                        1.76634852e02,
                        5.66410518e00,
                        1.28448645e02,
                        2.49195394e02,
                        2.73250649e02,
                        7.70883882e00,
                        1.63778163e00,
                    ]
                )
                if self.meal_announce is not None:
                    state = state / norm_arr
                else:
                    state = state / norm_arr[:-2]
            if self.suppress_carbs:
                state[:3] = 0.0
            if self.limited_gt:
                state = np.array([state[3], self.calculate_iob()])
            return state
        return np.stack(return_arr)

    def avg_risk(self):
        return np.mean(self.env.risk_hist[max(self.state_hist, 288) :])

    def avg_magni_risk(self):
        return np.mean(self.env.magni_risk_hist[max(self.state_hist, 288) :])

    def glycemic_report(self):
        bg = np.array(self.env.BG_hist[max(self.state_hist, 288) :])
        ins = np.array(self.env.insulin_hist[max(self.state_hist, 288) :])
        hypo = (bg < 70).sum() / len(bg)
        hyper = (bg > 180).sum() / len(bg)
        euglycemic = 1 - (hypo + hyper)
        return bg, euglycemic, hypo, hyper, ins

    @property
    def in_use(self):
        if self.use_only_during_day and (
            self.env.time.hour > 20 or self.env.time.hour < 5
        ):
            return False

    def is_done(self):
        horizon_complete = False
        if self.horizon is not None:
            horizon_complete = self.t >= self.horizon
        #         logger.info('Blood glucose: {}'.format(self.env.BG_hist[-1]))
        return (
            self.env.BG_hist[-1] < self.reset_lim["lower_lim"]
            or self.env.BG_hist[-1] > self.reset_lim["upper_lim"]
            or horizon_complete
        )

    def increment_seed(self, incr=1):
        # if type(self.seeds) == Seed:
        #     seed = self.seeds
        #     self.seeds = {}
        #     self.seeds['numpy'] = seed.numpy_seed
        #     self.seeds['scenario'] = seed.scenario_seed
        #     self.seeds['sensor'] = seed.sensor_seed
        self.seeds["numpy"] += incr
        self.seeds["scenario"] += incr
        self.seeds["sensor"] += incr

    def reset(self):
        return self._reset()

    def set_patient_dependent_values(self, patient_name, noise_scale=1.0):
        self.patient_name = patient_name
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split("#")[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))[
            "BW"
        ].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))[
            "u2ss"
        ].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.0
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        if self.rolling_insulin_lim is not None:
            self.rolling_insulin_lim = (
                (self.rolling_insulin_lim * self.bw)
                / self.CR
                * self.rolling_insulin_lim
            ) / 5
        else:
            self.rolling_insulin_lim = None
        iob_all = joblib.load("{}/iob.pkl".format(self.pid_env_path))
        self.iob = iob_all[self.patient_name]
        pid_df = pd.read_csv(self.pid_para_file)
        if patient_name not in pid_df.name.values:
            raise ValueError("{} not in PID csv".format(patient_name))
        pid_params = pid_df.loc[pid_df.name == patient_name].squeeze()
        self.pid = pid.PID(
            setpoint=pid_params.setpoint,
            kp=pid_params.kp,
            ki=pid_params.ki,
            kd=pid_params.kd,
        )
        patient = T1DPatientNew.withName(patient_name, self.patient_para_file)
        sensor = CGMSensor.withName(
            "Dexcom",
            self.sensor_para_file,
            seed=self.seeds["sensor"],
            noise_scale=noise_scale,
        )
        if self.time_std is None:
            scenario = RandomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds["scenario"],
                kind=self.kind,
                restricted=self.restricted_carb,
                harrison_benedict=self.harrison_benedict,
                unrealistic=self.unrealistic,
                deterministic_meal_size=self.deterministic_meal_size,
                deterministic_meal_time=self.deterministic_meal_time,
                deterministic_meal_occurrence=self.deterministic_meal_occurrence,
                meal_duration=self.meal_duration,
            )
        elif self.use_custom_meal:
            scenario = CustomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds["scenario"],
                num_meals=self.custom_meal_num,
                size_mult=self.custom_meal_size,
            )
        else:
            scenario = SemiRandomBalancedScenario(
                bw=self.bw,
                start_time=self.start_time,
                seed=self.seeds["scenario"],
                time_std_multiplier=self.time_std,
                kind=self.kind,
                harrison_benedict=self.harrison_benedict,
                meal_duration=self.meal_duration,
            )
        pump = InsulinPump.withName("Insulet", self.insulin_pump_para_file)
        self.env = T1DSimEnv(
            patient=patient,
            sensor=sensor,
            pump=pump,
            scenario=scenario,
            sample_time=self.sample_time,
            source_dir=self.source_dir,
        )
        if self.hist_init:
            self.env_init_dict = joblib.load(
                "{}/{}_data.pkl".format(self.pid_env_path, self.patient_name)
            )
            self.env_init_dict["magni_risk_hist"] = []
            for bg in self.env_init_dict["bg_hist"]:
                self.env_init_dict["magni_risk_hist"].append(magni_risk_index([bg]))
            self._hist_init()

    def _reset(self):
        self.t = 0
        if self.update_seed_on_reset:
            self.increment_seed()
        if self.use_model:
            if self.load:
                self.env = joblib.load(
                    "{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name)
                )
                self.env.model = self.model
                self.env.model_device = self.model_device
                self.env.norm_params = self.norm_params
                self.env.state = self.env.patient.state
                self.env.scenario.kind = self.kind
            else:
                self.env.reset()
        else:
            if self.load:
                if self.use_old_patient_env:
                    self.env = joblib.load(
                        "{}/{}_env.pkl".format(self.pid_env_path, self.patient_name)
                    )
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                else:
                    self.env = joblib.load(
                        "{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name)
                    )
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                if self.time_std is not None:
                    self.env.scenario = SemiRandomBalancedScenario(
                        bw=self.bw,
                        start_time=self.start_time,
                        seed=self.seeds["scenario"],
                        time_std_multiplier=self.time_std,
                        kind=self.kind,
                        harrison_benedict=self.harrison_benedict,
                        meal_duration=self.meal_duration,
                    )
                self.env.sensor.seed = self.seeds["sensor"]
                self.env.scenario.seed = self.seeds["scenario"]
                self.env.scenario.day = 0
                self.env.scenario.weekly = self.weekly
                self.env.scenario.kind = self.kind
            else:
                if self.universal:
                    patient_name = np.random.choice(self.universe)
                    self.set_patient_dependent_values(patient_name)
                self.env.sensor.seed = self.seeds["sensor"]
                self.env.scenario.seed = self.seeds["scenario"]
                self.env.reset()
                self.pid.reset()
                if self.use_pid_load:
                    self.pid_load(1)
                if self.hist_init:
                    self._hist_init()
        return self.get_state(self.norm)

    def _hist_init(self):
        self.rolling = []
        env_init_dict = copy.deepcopy(self.env_init_dict)
        self.env.patient._state = env_init_dict["state"]
        self.env.patient._t = env_init_dict["time"]
        if self.start_date is not None:
            # need to reset date in start time
            orig_start_time = env_init_dict["time_hist"][0]
            new_start_time = datetime(
                year=self.start_date.year,
                month=self.start_date.month,
                day=self.start_date.day,
            )
            new_time_hist = (
                (np.array(env_init_dict["time_hist"]) - orig_start_time)
                + new_start_time
            ).tolist()
            self.env.time_hist = new_time_hist
        else:
            self.env.time_hist = env_init_dict["time_hist"]
        self.env.BG_hist = env_init_dict["bg_hist"]
        self.env.CGM_hist = env_init_dict["cgm_hist"]
        self.env.risk_hist = env_init_dict["risk_hist"]
        self.env.LBGI_hist = env_init_dict["lbgi_hist"]
        self.env.HBGI_hist = env_init_dict["hbgi_hist"]
        self.env.CHO_hist = env_init_dict["cho_hist"]
        self.env.insulin_hist = env_init_dict["insulin_hist"]
        self.env.magni_risk_hist = env_init_dict["magni_risk_hist"]

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    @property
    def action_space(self):
        return spaces.Box(low=0, high=0.1, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
        if self.gt:
            return spaces.Box(low=0, high=np.inf, shape=(len(st),), dtype=np.float64)
        else:
            num_channels = int(np.prod(st.shape) / self.state_hist)
            return spaces.Box(
                low=0,
                high=np.inf,
                shape=(num_channels, self.state_hist),
                dtype=np.float64,
            )


register_env("glucose_env", lambda config: SimglucoseEnv(config))
register_env(
    "glucose_env_multiagent", make_multi_agent(lambda config: SimglucoseEnv(config))
)
