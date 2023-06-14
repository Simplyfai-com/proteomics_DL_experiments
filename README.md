# proteomics_DL_experiments

Tasks
  Retention time prediction 
  Fragment intensity prediction 
  
Retention time prediction

  Dataset - https://figshare.com/articles/dataset/ProteomeTools_-_Prosit_iRT_-_Data_update_missing_c_splitting/10043132

  Model  - Vanilla transformer style architecture 

  Evaluation metric - pearson correlation coefficient
  
Fragment intensity prediction
  
  Dataset - https://figshare.com/articles/dataset/LMDB_data_Tape_Input_Files/16688905
  
  Model consists of three modules 

    Peptide sequence encoder - initialized with the model used for the Task 1 and fine tuned on the Fragment intensity prediction dataset
    Collision energy and charge state latent embedding is fused with the peptide sequence latent features using shift and scale operations as opposed to the scale only method used in the prosit paper 
    Learned latent arrays are used to model the break points of a peptide sequence and they attend to the modulated peptide sequence features using cross attention method applied iteratively. This can be regarded as performing fully end to end clustering of the modulated peptide features with the learned latent vectors as the clustering centers.
    
  Evaluation metric: angular distance 

	

