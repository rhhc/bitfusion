# BitFusion Simulator

See bitfusion-generate-graphs.ipynb for details on how to use the simulator.

# Usage

  1. install
      ```bash
      git clone https://github.com/rhhc/bitfusion.git
      cd bitfusion/bitfusion/sram
      git clone https://github.com/HewlettPackard/cacti
      cd cacti
      make all -j32
      cd ../../..
      pip install -r requirements.txt
      ```
  2. profiling
     ```bash
     # screen -S bitfusion_$index
     # export bitfusion_index=resnet-index
     # export bitfusion_thread=50
     # set the en
     export PYTHONPATH=$PYTHONPATH:`pwd`
     python main.py
     # result save in result/layer-wise-$index.csv
     ```
