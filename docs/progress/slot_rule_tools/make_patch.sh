#!/bin/bash
set -e
S=/private/tmp/claude-501/-Users-josephremingstonl-Downloads-code-capstone/2d91e859-4acd-4fbf-99c1-146f2e2aa197/scratchpad/f6rework; D=$S/slot-rule-v2; cd $D
rm -rf cmp; mkdir -p cmp/a cmp/b
CODE="main/retrieval/generation.py main/retrieval/rag.py main/__main__.py tests/test_abstention.py"
DATA="$(cd repo && ls evaluation/datasets/seeds/*.json) evaluation/datasets/gold_queries.json"
for f in $CODE $DATA; do mkdir -p cmp/a/$(dirname $f) cmp/b/$(dirname $f); [ -f $S/snapshot/$f ] && cp $S/snapshot/$f cmp/a/$f; cp repo/$f cmp/b/$f; done
cd cmp; : > ../F6-slot-rule-v2.code.patch
for f in $CODE; do diff -uN a/$f b/$f >> ../F6-slot-rule-v2.code.patch || true; done
cp ../F6-slot-rule-v2.code.patch ../F6-slot-rule-v2.full.patch
for f in $DATA; do diff -uN a/$f b/$f >> ../F6-slot-rule-v2.full.patch || true; done
cd ..; wc -l F6-slot-rule-v2.*.patch; grep -c "^+++" F6-slot-rule-v2.full.patch
rsync -a --exclude data $S/snapshot/ cmp/apply/ && cd cmp/apply && patch -p1 -s < ../../F6-slot-rule-v2.full.patch
for f in $CODE evaluation/datasets/gold_queries.json; do cmp -s $f ../../repo/$f && echo "same $f"; done
