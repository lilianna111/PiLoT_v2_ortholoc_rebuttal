Run example:

python test.py \
  --dsm_path /path/to/dsm_wgs84.tif \
  --dom_path /path/to/dom_wgs84.tif \
  --output_dir /path/to/output \
  --name sample \
  --lon 114.04270934472524 \
  --lat 22.416065873493764 \
  --alt 141.49985356855902 \
  --roll 4.551786987694334 \
  --pitch 1.3641392882272305 \
  --yaw 103.15503822838967 \
  --K_json ./K.json

Outputs:
- <name>_dom.png
- <name>_dsm.npy
- <name>_debug_overlay.png



python /home/amax/Documents/code/PiLoT/PiLoT/crop/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dsm_wgs84.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dom_wgs84.tif   --output_dir /home/amax/Documents/code/PiLoT/PiLoT/crop/test   --name sample   --lon 44.89959469169528   --lat 39.94467559552912   --alt 1368.9175022606412   --roll -4.4838410785042555   --pitch -1.7736037422911934   --yaw -1.673979757889432   --K_json /home/amax/Documents/code/PiLoT/PiLoT/crop/K.json

44.89959469169528 39.94467559552912 1368.9175022606412  -4.4838410785042555 -1.7736037422911934 -1.673979757889432