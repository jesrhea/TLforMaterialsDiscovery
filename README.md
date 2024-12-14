# Transfer Learning for Materials Discovery

This repository contains scripts written for the Master's project <i>Exploring the Use of Transfer Learning for Property Prediction of Metal and Covalent Organic Frameworks</i>:

- COF Gas Adsorption 

This folder contains scripts and data required for COFid generation and generating a representative subset of the original [COF gas adsorption dataset](https://archive.materialscloud.org/record/2018.0003/v2). 

-  MOF Thermal Conductivity 

This folder contains scripts to extract structural features and train tree-based models on the [MOF thermal conductivity dataset](https://github.com/meiirbek-islamov/thermal-transport-MOFs). There are also structural and MOFid data for fine-tuning MOFormer. 

## Abstract

The adoption of cleaner fuels to replace oil and gas is essential in reducing global greenhouse gas emissions and combatting climate change. Alternative fuels such as hydrogen and methane are gaseous and therefore challenging to store safely and efficiently. Metal organic frameworks (MOFs) and covalent organic frameworks (COFs) are two promising materials for gas storage applications, due to their light weight, large storage capacities and readily tuneable structures. Machine learning models can accelerate the discovery of novel MOFs and COFs by generating predictions of properties important in gas storage for large numbers of structures, faster and more cost-effectively than experiment or computational simulations. However, this machine learning-based discovery is hindered by the lack of available MOF and COF structure and property data.  Here transfer learning is used to overcome data scarcity in MOF thermal conductivity and COF gas adsorption data. The effectiveness of transfer learning is found to be limited by biases in the training data, and the complexity of the structure-property relationship to model. Though ineffective for MOF thermal conductivity prediction, transfer learning did improve the accuracy of COF gas adsorption predictions by up to 9.6%. This is significant, as it shows that more abundant MOF data can be leveraged for the discovery of optimised COFs. Overall, this project serves as an initial proof-of-concept study demonstrating the successes and challenges of machine learning and transfer learning in the discovery of novel MOFs and COFs optimised for gas storage applications. 

![image](https://github.com/user-attachments/assets/5e9022be-0745-4ea1-927e-ba4f7be47b07)
<i> Outline of project work. The MOF-TC and COF-GA datasets will be used to fine-tune MOFormer for MOF thermal conductivity and COF gas adsorption predictions respectively. An additional tree-based model will be trained directly on the MOF-TC dataset for MOF thermal conductivity prediction, serving as a benchmark for the fine-tuned MOFormer model. </i>
