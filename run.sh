#!/usr/bin/env bash
# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
# Start the route picker GUI at http://127.0.0.1:8777
cd "$(dirname "$0")"
python3 -c "import numpy, scipy, shapely, pyproj, PIL" 2>/dev/null || pip install --user -r requirements.txt
exec python3 server.py "$@"
