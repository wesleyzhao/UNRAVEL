# UNRAVEL
#### UN-biased high-Resolution Analysis and Validation of Ensembles using Light sheet images


UNRAVEL is a command line tool for:
* voxel-wise analysis of fluorescent signals (e.g., c-Fos immunofluorescence) across the mouse brains in atlas space
* validation of hot/cold spots via c-Fos+ cell density quantification and montages at cellular resolution
    
[UNRAVEL guide:](https://office365stanford-my.sharepoint.com/:p:/g/personal/danrijs_stanford_edu/EbQN54e7SwRHgkmw3yn8fgcBz1xG22AICtZx8nsPrOLFtg?e=S159PM)
* notes dependencies, paths to update, organization of files/folders, and info on running scripts:
* Key scripts: find_clusters.sh, glm.sh, and validate_clusters2.sh
* Scripts can be run in a modular fashion:
 + overview.sh -> prep_tifs.sh or czi_to_tif.sh -> 488_to_nii.sh -> reg.sh -> rb.sh -> z_brain_template_mask.sh -> glm.sh -> validate_clusters2.sh 
* Each script starts with a help guide. View by running: <script>.sh help 

If you are unfamiliar with the command line interface, please review Unix tutorials: https://andysbrainbook.readthedocs.io/en/latest/index.html

[Heifets lab guide to immunofluorescence staining, iDISCO+, & lightsheet fluorescence microscopy](https://docs.google.com/document/d/16yowBhiBQWz8_VX2t9Rf6Xo3Ub4YPYD6qeJP6vJo6P4/edit?usp=sharing)

Please send questions/suggestions to:
* Daniel Ryskamp Rijsketic (danrijs@stanford.edu)
* Austen Casey (abcasey@stanford.edu)
* Boris Heifets (bheifets@stanford.edu)
