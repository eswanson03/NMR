This code is part of the 2023 Nuclear Physics Eastern Tennessee research project. With signal strength information from NMR spectra data and in-lab material quantification,  
the exact amount of protons in deuterated poly-methyl methacrylate (dPMMA) can be identified. This is important to minimize the statistical uncertainty 
for neutron-electric dipole measurements at the [nEDM project] (https://nedm.ornl.gov/) at the Spallation Neutron Source. dPMMA is a material used in the heart of the experiment.

In the discussion of the scripts, two techniques are often mentioned: 
(1) 'Frequency-binning' - the process of slicing the entire frequency domain of the FFT into bins which can be examined. 
(2) 'Time-binning' - the process of slicing the entire time domain of the FFT into bins which can also be examined. 

Included are two scripts: 
(1) 'timeDependentSignalStrength_main' - A method of finding the initial signal strength of a molecular site without time binning the FFT domain. 
(2) 'timeIndependentSignalStrength_main' - A method of finding the signal stregth of a molecular site by time and frequency binning the FFT domain. 

In our experimental method, we created various in-lab samples to accurately quantify the amount of protons in dPMMA:
 (1) 'pPMMA sample' - A sample of known proton content which can be used as a standard
 (2) 'TB-MORE sample' - A sample of deuterated toluene (99.7%) and Benzene at a known ratio and proton content, also used as a standard. 
 (3) 'TB-LESS sample' - The same as sample (2), but the amount of toluene and Benzene is less. This allows us to compare the signal strength ratios to proton content ratio. 
 (4) 'dPMMA sample' - Our target sample with dissolved dPMMA. 

 Each sample has 7 corresponding FFT files. This is to allow a systematic quantification of the uncertainty associated with the NMR machine itself. 

The code is input based, asking the user to manually select the regions of interest and (if using the time dependent method) the desired time binning width. 

The result of this research resulted in a dPMMA polymer sample with a quantified proton content percentage. 
