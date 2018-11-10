# Architecture

The main file (vSomething.py) will be as limited as possible.
 - The timers will be defined here.
 - The signal/slot functions will be specified here, but the slot functions will not be contained here.
 - The function to step through the scanning procedure
 - The function that saves the data. It should provide ability to save in a sane dataformat (hdf5) or something that's compatible with the OpenWave software.
 - A function to save the relevant parameters to the .yaml file. A second step for this function would be to generate it, if it was lost.
 - A function to update the status window and status/progress bar.


A file (diffPlotting.py) will take care of the plotting:
 - A function to update the raw feed and the difference feed.
 - A function to update the histograms.
 - The function to specify the colormap.


A file (stageController.py) will take care of the stage communications.
 - A function to home the stage on start. (Note: I need to decide on a decent time to do that)
 - A function to move to a specified position


A .yaml file to keep the non-volatile memory of the settings.


The status window should be considered. The aim is that it will contain information about:

 - Acquisition status (Running, )
 - Estimated remaining time of acq.
 - The connected camera
 - The connected stage 
