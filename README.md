# TAQADataset

About the Teacher Action Quality Assessment Dataset（TAQA）dataset
-----------------


The dataset contains 4 main categories of teachers' frequent teaching actions in the teaching process. There are a total of 3998 video samples, covering 9 secondary school subjects. The details are shown in Figure 1.
![Figure 1: An Overview and distribution of TAQA dataset.](https://github.com/aauthorsanonymous/TAQADataset/assets/142662941/e0d09d2a-e746-4af3-b509-546fc54b18a8)
                            Figure 1: An Overview and distribution of TAQA dataset.

**1.Action_type** represents the action category, and the dataset contains a total of four actions, namely Pointing to teaching devices, Asking students to answer questions, Tour guide and Blackboard-writing. 

**2.#Samples** represents the number of samples in each category. TAQA dataset includes 1215 “Blackboard-writing” samples, 589 “Tour guide” samples, 1132 “Pointing to teaching devices” samples and 1062 “Asking students to answer questions” samples.

**3.Avg.Seq.Len** represents the average number of video frames for each category. Avg.Seq.Len of “Blackboard-writing” is 591, Avg.Seq.Len of “Tour guide” is 595, Avg.Seq.Len of “Pointing to teaching devices” is 261，Avg.Seq.Len of “Asking students to answer questions” is 119.

**4.View Variation/Background** indicates whether the action video has a change in perspective or background. “Blackboard-writing”, “Pointing to teaching devices” and “Asking students to answer questions” have the same perspective or background. Tour guide has the same background and a little perspective change.

**5.Judge_scores** indicate the score of teacher action by each education expert. Final_score is the final score of the action video.

**The detailed partition of training set and test set is given in our paper.**



About the Teacher Action Assessment Model (TAAM) model
-------------------------------------------------------

### Requirement
 
   
- Python >= 3.6
- Pytorch >=1.8.0
 
 
### Dataset Preparation
#### TAQA dataset
 
If the article is accepted for publication, you can download our prepared TAQA dataset demo from [Google Drive](https://drive.google.com/drive/folders/1UaqtLGsA_pMyOZvoh7IG4IVM9K8siYot?usp=sharing). Then, please move the uncompressed data folder to `TAQA/data/frames`. We used the I3D backbone pretrained on Kinetics([Google Drive](https://drive.google.com/open?id=1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU)).
 
#### Training & Evaluation
To train and evaluate the TAAM model on TAQA:

`python -u main.py  --lr 1e-4 --weight_decay 1e-5 --gpu 0`



If you use the TAQA dataset, please cite this paper:
A Teacher Action Assessment Model based on a New Assessment Dataset.
