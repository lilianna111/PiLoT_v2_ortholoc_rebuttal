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



python /home/amax/Documents/code/PiLoT/PiLoT/crop/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/DSM.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/DOM.tif   --output_dir /home/amax/Documents/code/PiLoT/PiLoT/crop/test   --name sample   --lon 44.89959469169528   --lat 39.94467559552912   --alt 1368.9175022606412   --roll -4.4838410785042555   --pitch -1.7736037422911934   --yaw -1.673979757889432   --K_json /home/amax/Documents/code/PiLoT/PiLoT/crop/K.json

44.89959469169528 39.94467559552912 1368.9175022606412  -4.4838410785042555 -1.7736037422911934 -1.673979757889432
114.04270934472524 22.416065873493764 141.49985356855902  4.551786987694334 1.3641392882272305 103.15503822838967


python /home/amax/Documents/code/PiLoT_v2/crop/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dsm_wgs84.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dom_wgs84.tif   --output_dir /home/amax/Documents/code/PiLoT_v2/crop/test   --name sample   --lon 114.04270934472524   --lat 22.416065873493764   --alt 141.49985356855902   --roll 4.551786987694334   --pitch 1.3641392882272305   --yaw 103.15503822838967   --K_json /home/amax/Documents/code/PiLoT_v2/crop/K.json



python /home/amax/Documents/code/PiLoT_v2/crop/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKisland_GNSS/dsm_wgs84.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKisland_GNSS/dom_wgs84.tif   --output_dir /home/amax/Documents/code/PiLoT_v2/crop/test   --name sample   --lon 114.25834288619072   --lat 22.205296022324383   --alt 100.75537325828935   --roll 1.4894249861234055   --pitch 2.1523655622721525   --yaw 67.25778027171904   --K_json /home/amax/Documents/code/PiLoT_v2/crop/K.json


1698132142.199930191.jpg 114.25856659002368 22.205075272773776 100.56471835349186  -6.979374889230779 0.8059505648727494 -176.4363512569613
114.25834288619072 22.205296022324383 100.75537325828935  1.4894249861234055 2.1523655622721525 67.25778027171904

python /home/amax/Documents/code/PiLoT_v2/crop/test.py \
  --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/mapscape/model/usa_2/dsm.tif \
  --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/mapscape/model/usa_2/dom.tif \
  --output_dir /home/amax/Documents/code/PiLoT_v2/tmp_crop_out \
  --name test1 \
  --lon -87.621232 \
  --lat 41.860991999999996 \
  --alt 500.0 \
  --roll 0.0 \
  --pitch 10.0 \
  --yaw 281.3096633649805 \
  --K_json /home/amax/Documents/code/PiLoT_v2/crop_wgs84_0329/K_mapscape.json 