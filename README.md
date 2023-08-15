# A Pipeline to Evaluate the Effects of Noise on Machine Learning Detection of Laryngeal Cancer

This work has been accepted into Interspeech 2023. The full paper can be found [here](https://www.isca-speech.org/archive/interspeech_2023/paterson23_interspeech.html).

In this paper we describe a pipeline for the evaluation of classification models in the presence of noise. We demonstrate this pipeline using four classification models and three denoising techniques. For these experiments we use the Saarbruecken Voice Database (found [here](https://stimmdatenbank.coli.uni-saarland.de/help_en.php4#menu)). 

All four models created for the experiments can be found in the 'Models' folder. The results2.csv file contains the predictions made for each file in the test set using each of the four models as well as the demographic information for each recording. 

Figure 2 can be generated using the notebook titled 'AccuracyPerNoise'

Figure 3 can be generated using the notebook titled 'CompareDenoising'
 
