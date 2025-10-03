#!/bin/bash

git fetch origin main          # обновить знание о том, что на сервере
git push --force-with-lease=main:$(git rev-parse origin/main) origin HEAD:main
git add -A && git commit -m "sync local -> remote" || true && git push -u origin main --force-with-lease


#cd ~/rknpu2/examples/calib
#git init
#git add -A
#git commit -m "Initial commit from local"
#git branch -M main
#git remote add origin git@github.com:ilyamus74-ctrl/make_multi_2probe.git
#git push -u origin main