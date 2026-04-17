#!/bin/bash

# WebArena service endpoints.
# These must match the ports exposed on the current node (SSH tunnel from
# 69.157.137.231 → localhost, see Eval/evaluate_sft_webrl.sh for the tunnel
# command). If you change the tunnel, change these.

PUBLIC_HOSTNAME="localhost"

export DATASET="webarena"

# CLASSIFIEDS is not used in WebArena-Lite but must be defined to avoid KeyError
# during env init.
export CLASSIFIEDS="http://localhost:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

SHOPPING_PORT=7770
SHOPPING_ADMIN_PORT=7780
REDDIT_PORT=9999
GITLAB_PORT=8023
WIKIPEDIA_PORT=8888
HOMEPAGE_PORT=4399
MAP_PORT=3000

export SHOPPING="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export REDDIT="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}"
export GITLAB="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}"
export WIKIPEDIA="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://${PUBLIC_HOSTNAME}:${HOMEPAGE_PORT}"
export MAP="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"

export OPENAI_API_KEY="xxxx"
export OPENAI_API_URL="https://api.openai.com/v1"
