trap ctrl_c INT
docker run -v "$PWD":/workspace fl-edge-simulation "$@"
