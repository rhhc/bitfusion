# bitfusion
Simulator for BitFusion

See bitfusion-generate-graphs.ipynb for details on how to use the simulator.

# Usage

  1. install
      ```
      project=/workspace/git  # change to your own project folder
      cd $project
      git clone https://github.com/hsharma35/dnnweaver2
      git clone https://github.com/blueardour/bitfusion
      cd bitfusion
      
      ## change to python version 2 
      # pyenv local 2.7.16
      python -V  # should be version 2
      
      pip install -r requirements.txt
      ```
  2. profiling
     ```
     #index in range(51)
     screen -S bitfusion_$index
     export bitfusion_index=resnet-index
     export bitfusion_thread=50
     python main.py
     # result save in result/layer-wise-$index.csv
     ```
