1. Your name and any other group members, if any.
Yunxuan 'Kaya' Rao

2. Links/URLs to any videos you created and want to submit as part of your report.
No link

3. What operating system and IDE you used to run and compile your code.
VsCode + MacOS

4. Instructions for running your executables.  
Run "./FeatureExtraction <image_directory> <output_csv_file> <distance_matrix>" First
Then Run "./ImgRetrieval <target_image> <feature_csv_file> <N> <distance_matrix>"

For dnn method, only run ./ImgRetrieval
For <distance_matrix>:
Q1 - baseline
Q2 - histogram
Q3 - multihist
Q4 - tc
Q5 - dnn

The files are structured like below:
src
│
├── featureExtraction.cpp  
├── imgRetrieval.cpp    
├── CMakeLists.txt          
└── matchingMethods
    │
    ├── baselineMatching.cpp
    ├── baselineMatching.h
    ├── deepNetworkMatching.cpp 
    ├── deepNetworkMatching.h    
    ├── histogramMatching.cpp   
    ├── histogramMatching.h
    ├── multiHistogramMatching.cpp  
    ├── multiHistogramMatching.h
    ├── textureColor.cpp         
    └── textureColor.h          
    
    

5. Instructions for testing any extensions you completed.
No


6. Whether you are using any time travel days and how many.
2 days