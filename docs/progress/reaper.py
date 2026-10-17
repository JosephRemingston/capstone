"""Kill leftover processes of workflow agents that have finished (their result is in the journal)."""
import json, subprocess, time
from pathlib import Path

WF = Path.home() / '.claude/projects/-Users-josephremingstonl-Downloads-code-capstone/2d91e859-4acd-4fbf-99c1-146f2e2aa197/subagents/workflows'
S = '/private/tmp/claude-501/-Users-josephremingstonl-Downloads-code-capstone/2d91e859-4acd-4fbf-99c1-146f2e2aa197/scratchpad'
# agent label -> scratch directory its jobs run in
DIRS = {
    'verify:rerank-graph-time': f'{S}/design/verify-rerank-graph-time',
    'verify:ann-and-pool': f'{S}/design/verify-ann-and-pool',
    'author:unanswerable+calibration': f'{S}/f6rework/author',
    'design:slot-rule-v2': f'{S}/f6rework/slot-rule-v2',
    'design:answerability-model': f'{S}/f6rework/answerability-model',
    'verify:slot-rule-v2': f'{S}/f6rework/verify-slot-rule-v2',
    'verify:answerability-model': f'{S}/f6rework/verify-answerability-model',
}

def finished_labels():
    done = set()
    for journal in WF.glob('*/journal.jsonl'):
        started = {}
        for line in journal.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('type') == 'started':
                started[e.get('agentId')] = e.get('label')
            elif e.get('type') == 'result' and e.get('agentId') in started:
                done.add(started[e['agentId']])
    return done

def python_procs():
    out = subprocess.run(['ps', '-eo', 'pid=,command='], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(' ')
        if 'python' in cmd.lower() and 'reaper.py' not in cmd:
            cwd = subprocess.run(['lsof', '-a', '-p', pid, '-d', 'cwd', '-Fn'], capture_output=True, text=True).stdout
            path = next((l[1:] for l in cwd.splitlines() if l.startswith('n')), '')
            yield int(pid), path, cmd

while True:
    done = finished_labels()
    for pid, cwd, cmd in python_procs():
        for label, folder in DIRS.items():
            if label in done and cwd.startswith(folder + '/'):
                subprocess.run(['kill', str(pid)])
                print(time.strftime('%H:%M:%S'), 'killed', pid, label, cmd[:120], flush=True)
    time.sleep(30)
