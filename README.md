# Terahertz measurement setup for 2D measurements

## Background
See the publication by F. Blanchard, et al., 25 April 2011 / Vol. 19, No. 9 / OPTICS EXPRESS pg.8284.

## Electro Optical detection of THz
A second order process takes place in the ZnTe crystal when high intensity THz beams interacts with the electrical field of a probe beam. This generates a small amount of light positioned as sidebands with an energy difference from the probe equal to the THz photon energy. The minute difference would be tricky to measure, if not for the fact that the polarisation of the sidebands is normal to the probe. 

Two options are available; adding a crossed polarizer as an analyzer, and measure the contribution that is polarised normal to the probe. This can lead to spurious high frequency components, as a bit of the probe invariably leaks through.

The other option is to split the light with a Wollaston prism, and measure the difference between the two contributions, similar to the well-known setup with the balanced photodiodes.

This software should allow the latter. Point both beams from the Wollaston prism onto a CCD/CMOS in the image plane. 

Note that the integration of the camera is very raw so far. It's just been developed with the image feed from the webcam on my computer, and an 8bit image is assumed.

## Missing bits
The stage control is not implemented, and neither is the file management tools. Presently it's very rough, but the GUI is laid out, and the camera subtraction is functional.
