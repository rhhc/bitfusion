# BitFusion Simulator

See bitfusion-generate-graphs.ipynb for details on how to use the simulator.

1. Install
 
```bash
 git clone --recursive https://github.com/rhhc/bitfusion.git
 cd bitfusion/bitfusion/sram/cacti
 make all -j32
 cd ../../..
 pip install -r requirements.txt
```

2. Profiling

 ```bash
 source setup.sh
 ./scripts/generate_latency_by_layer.sh
 ```
