"""Reproduce the paired evaluation of serve-return and later-return policies.

Only the evaluator binds current public point history to a policy choice.
The original physics, service/recovery control, actor inputs and replay remain.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from simtennis.rally import evaluation
from simtennis.rally.environment import RallyConfig
from simtennis.rally.policy import FrozenPolicy
from simtennis.rally.practice import PracticeConfig, ReturnPracticeEnv, evaluate_practice


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checkpoint_identity(path):
    path = Path(path)
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def receiving_service(world,player):
    return (world.setup.get('kind')=='serve' and world.incoming(player)
            and sum(world.rules.racket_hits)==1)


@dataclass(frozen=True)
class Specialists:
    service: object
    rally: object

    def select(self,world,player):
        return self.service if receiving_service(world,player) else self.rally


def bound_points(counter):
    original=evaluation.play_point
    def call(world,policies,*args,**kwargs):
        def bind(policy,player):
            if not isinstance(policy,Specialists):return policy
            def action(observation):
                assert world.incoming(player)
                first=receiving_service(world,player)
                counter['service' if first else 'rally']+=1
                if first and not world.rules.serving:counter['service_after_bounce']+=1
                return policy.select(world,player)(observation)
            return action
        return original(world,[bind(policy,p) for p,policy in enumerate(policies)],*args,**kwargs)
    return patch.object(evaluation,'play_point',call)


def qualify(base):
    counter=Counter();rows=[]
    for opening in (0,1):
        plain=evaluation.play_contest([base,base],96100+opening*5,first_server=opening,ends=opening)
        with bound_points(counter):
            routed=evaluation.play_contest([Specialists(base,base)]*2,96100+opening*5,
                                           first_server=opening,ends=opening)
        # This compares every action, rendered pose, contact and score; wall time
        # is the sole excluded field. Both roles and both physical ends occur.
        assert {k:v for k,v in plain.items() if k!='seconds'}=={k:v for k,v in routed.items() if k!='seconds'}
        rows.append({'opening_server':opening,'ends':opening,'points':plain['points'],
                     'native_controls':sum(e['steps'] for e in plain['episodes'])})
    assert counter['service_after_bounce']>0 and counter['rally']>0
    return {'identical_actor_routing':rows,'calls':dict(counter)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    bundle = ROOT/'demo'
    parser.add_argument('--base', type=Path, default=bundle/'serve-return.zip')
    parser.add_argument('--later', type=Path, default=bundle/'later-return.zip')
    parser.add_argument('--opponent', type=Path, action='append', help='Repeat in evaluation order; defaults to the bundled historical and serve-return policies')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1);started=time.perf_counter()
    base_path=args.base
    paths=args.opponent or [bundle/'historical-opponent.zip',base_path]
    base=FrozenPolicy.load(base_path);later=FrozenPolicy.load(args.later)
    opponents=[FrozenPolicy.load(path) for path in paths]
    candidate=Specialists(base,later)
    manifest={'source':checkpoint_identity(base_path),'later':checkpoint_identity(args.later),
              'opponents':[checkpoint_identity(path) for path in paths],
              'fresh_practice_bank_seed':81600,'fresh_practice_bank_size':32,
              'contest_seeds':list(range(96200,96208)),
              'routing':'Native public point history: serve setup, incoming, exactly one accepted racket strike. Persists through service bounce.',
              'selection':'later-specialist-02 final preselected on development8/19 vs incumbent4/19',
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(args.output/'manifest.json',manifest);shutil.copy2(__file__,args.output/'driver.py')
    atomic_json(args.output/'qualification.json',qualify(base))
    config=RallyConfig(contact_reward=1,return_reward=10,win_reward=2)
    env=ReturnPracticeEnv(opponents,config,PracticeConfig(bank_seed=81600,bank_size=32,
                         kind='mixed',lead_seconds=4,miss_distance_scale=1))
    try:
        check={}
        for name,policy in [('incumbent',base),('candidate',candidate)]:
            def routed(obs):
                return policy.select(env.world,env.learner)(obs) if isinstance(policy,Specialists) else policy(obs)
            check[name]=evaluate_practice(routed,opponents,env=env,seeds=range(97600,97600+len(env.bank)))
        atomic_json(args.output/'fresh-practice.json',check)
    finally:env.close()
    contest_results={};counter=Counter()
    with bound_points(counter):
        for name,policy in [('incumbent',base),('candidate',candidate)]:
            contest_results[name]=evaluation.evaluate_contests(policy,opponents,manifest['contest_seeds'])
    atomic_json(args.output/'fresh-contests.json',contest_results)
    result={'seconds':time.perf_counter()-started,'routing_calls':dict(counter),
        'practice':{name:{'successes':r['successes'],'count':r['count'],'by_kind':r['by_kind'],
            'by_turn':{str(turn):{'cases':sum(e['incoming_turn']==turn for e in r['episodes']),
                'returns':sum(e['incoming_turn']==turn and e['is_success'] for e in r['episodes'])}
                for turn in sorted({e['incoming_turn'] for e in r['episodes']})}} for name,r in check.items()},
        'contests':{name:{k:r[k] for k in ('contest_wins','incomplete_contests','points','wins','rally_length_counts','by_opponent','by_role')}
                    for name,r in contest_results.items()}}
    atomic_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
