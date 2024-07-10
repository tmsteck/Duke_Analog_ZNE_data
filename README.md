This folder includes the scripts and file management for processing the results from Henry Luo's experimental runs at Duke. 

Experimental data from Rabi Oscillations go in a labeled folder in the data/ folder. Experiment IDs and waiting times can be dumped in a metadata.txt file

The script "Full_calibration.ipynb" returns updated t_w times and the amplitude compensation terms. 

It require manually entering the experiment ids (in order) and adding the waiting times as a list. 

When running the script, be sure to refresh the kernel. Also, if it is the first time running a data set, set the foldername, and enter 'y' into the first prompt, and add any useful experimental info when prompted for metadata. this is purely for record keeping. 
