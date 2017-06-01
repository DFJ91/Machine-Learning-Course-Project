# Machine Learning Course Project
David Fernandez Jimenez  
29/5/2017  

## Executive Summary

The aim of this project is to develop a machine learning algorithm in order to be able to predict if the subject is lifting the barbell according to the specification or not. In case he/she is committing any mistake when lifting the barbell this would be classified as well in four different types.

## Summary of the data and a basic exploratory data analysis

The dataset collects the measurements of four sensors distributed among the body of the person who is doing the barbell lifting and the dumbbell itself. Three of them calculate the orientation of the arm, the forearm and the belt and last one calculates the orientation of the dumbbell. According to these values the activity would be classified in five ways: according to the specification (A), throwing the elbows to the front (B), lifting the dumbbell only halfway (C), lowering the dumbbell only halfway (D) and throwing the hips to the front (E).

In order to perform some basic exploratory data analyses I am going to process this data so all the information available would be easier to 
understand:

```r
training_data<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

testing_data<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

set.seed(32323)

library(caret)

#head(training_data)

#head(testing_data)

#names(training_data)==names(testing_data)
```
At first sight we can see in the appendix that both training dataset and testing datasets have a great number of variables which value is NA or missing. The next step would be to eliminate all those variable as they do not provide important information. Before doing that it has been checked that these variables has almost the same values in the testing dataset.


```r
training_data_2<-training_data

testing_data_2<-testing_data

#Removing the variables with NA values

training_data_2<-training_data_2[ , ! apply( training_data_2 , 2 , function(x) any(is.na(x)) ) ]

#Removing the variables with no values
i<-1

a<-0

for(j in 1:ncol(training_data_2)){
        if (as.character(training_data_2[1,j]) == "") {a[i]<-j ;i<-i+1}
}

training_data_2<-training_data_2[ , -c(a)]

#Removing the first 7 column of the original dataset as they are not
#going to be used develop the algorithm

training_data_2<-training_data_2[,-c(1,2,3,4,5,6,7)]

#Finally the testing dataset is modified to have the same variables as
#the training dataset

training_data_2_transf<-training_data_2[,!names(training_data_2) %in% c("classe")]

testing_data_2<-testing_data_2[,names(training_data_2_transf)]
```

## Model Selection

Once the training dataset and the testing dataset are ready the next will be to split the training dataset in six equal parts. This partition will provide three differents dataset to train the algorithm and other three different datasets to validate it.

```r
folds<-createFolds(y=training_data_2$classe,k=6,list = TRUE,returnTrain = FALSE)

sapply(folds,length)
```

```
## Fold1 Fold2 Fold3 Fold4 Fold5 Fold6 
##  3270  3272  3271  3270  3270  3269
```

```r
training1<-training_data_2[folds$Fold1,]

validation1<-training_data_2[folds$Fold2,]

training2<-training_data_2[folds$Fold3,]

validation2<-training_data_2[folds$Fold4,]

training3<-training_data_2[folds$Fold5,]

validation3<-training_data_2[folds$Fold6,]
```
During the course there have been seen a lot of different types of machine learning algorithms which would be possible to apply in this case. However there are two of them that highlights from the rest. Random Forests and Boosting are two of the most widely used and highly accurate methods for prediction. With this fact in mind, there are going to be trained three random forests algorithms and three boosting algorithms in order to average the results and determine which method is slightly best for this particular dataset.


```r
# Algorithm 1 Random Forest

modFit_RF_1<-train(classe~.,method="rf",data=training1)

pred1<-predict(modFit_RF_1,validation1)

# Algorithm 2 Random Forest

modFit_RF_2<-train(classe~.,method="rf",data=training2)

pred2<-predict(modFit_RF_2,validation2)

# Algorithm 3 Random Forest

modFit_RF_3<-train(classe~.,method="rf",data=training3)

pred3<-predict(modFit_RF_3,validation3)
```


```r
# Algorithm 1 Random Forest

confusionMatrix(pred1,validation1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 923  25   0   3   0
##          B   1 591   7   2   0
##          C   2  15 554  14   4
##          D   2   0  10 515   5
##          E   2   2   0   2 593
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9707          
##                  95% CI : (0.9643, 0.9762)
##     No Information Rate : 0.2842          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9629          
##  Mcnemar's Test P-Value : 2.328e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9925   0.9336   0.9702   0.9608   0.9850
## Specificity            0.9880   0.9962   0.9870   0.9938   0.9978
## Pos Pred Value         0.9706   0.9834   0.9406   0.9680   0.9900
## Neg Pred Value         0.9970   0.9843   0.9937   0.9923   0.9966
## Prevalence             0.2842   0.1935   0.1745   0.1638   0.1840
## Detection Rate         0.2821   0.1806   0.1693   0.1574   0.1812
## Detection Prevalence   0.2906   0.1837   0.1800   0.1626   0.1831
## Balanced Accuracy      0.9903   0.9649   0.9786   0.9773   0.9914
```

```r
# Algorithm 2 Random Forest

confusionMatrix(pred2,validation2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 919  15   1   1   0
##          B   5 607   4   6   6
##          C   5   3 560  15   5
##          D   1   4   5 508   1
##          E   0   4   0   6 589
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9734          
##                  95% CI : (0.9673, 0.9786)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9663          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9882   0.9589   0.9825   0.9478   0.9800
## Specificity            0.9927   0.9920   0.9896   0.9960   0.9963
## Pos Pred Value         0.9818   0.9666   0.9524   0.9788   0.9833
## Neg Pred Value         0.9953   0.9902   0.9963   0.9898   0.9955
## Prevalence             0.2844   0.1936   0.1743   0.1639   0.1838
## Detection Rate         0.2810   0.1856   0.1713   0.1554   0.1801
## Detection Prevalence   0.2862   0.1920   0.1798   0.1587   0.1832
## Balanced Accuracy      0.9905   0.9755   0.9860   0.9719   0.9881
```

```r
# Algorithm 3 Random Forest

confusionMatrix(pred3,validation3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 921  12   0   0   1
##          B   4 593  13   1   5
##          C   4  27 550  21   5
##          D   1   0   7 510   1
##          E   0   0   0   4 589
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9676          
##                  95% CI : (0.9609, 0.9734)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.959           
##  Mcnemar's Test P-Value : 0.0001405       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9903   0.9383   0.9649   0.9515   0.9800
## Specificity            0.9944   0.9913   0.9789   0.9967   0.9985
## Pos Pred Value         0.9861   0.9627   0.9061   0.9827   0.9933
## Neg Pred Value         0.9961   0.9853   0.9925   0.9905   0.9955
## Prevalence             0.2845   0.1933   0.1744   0.1640   0.1838
## Detection Rate         0.2817   0.1814   0.1682   0.1560   0.1802
## Detection Prevalence   0.2857   0.1884   0.1857   0.1588   0.1814
## Balanced Accuracy      0.9924   0.9648   0.9719   0.9741   0.9893
```

For the Random Forests algorithms the average accuracy is 0.968 with the worst 95% confidence interval of (0.957,0.970)

Let see now what are the results for the boosting algorithms:


```r
# Algorithm 1 Boosting
modFit_B_1<-train(classe~.,method="gbm",data=training1)

pred1_B<-predict(modFit_B_1,validation1)

# Algorithm 2 Boosting

modFit_B_2<-train(classe~.,method="gbm",data=training2)

pred2_B<-predict(modFit_B_2,validation2)


# Algorithm 3 Boosting

modFit_B_3<-train(classe~.,method="gbm",data=training3)

pred3_B<-predict(modFit_B_3,validation3)
```



```r
# Algorithm 1 Boosting

confusionMatrix(pred1_B,validation1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 912  25   2   0   1
##          B   9 576  12   0  10
##          C   6  26 544  24  11
##          D   2   1   9 498   7
##          E   1   5   4  14 573
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9483          
##                  95% CI : (0.9402, 0.9557)
##     No Information Rate : 0.2842          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9347          
##  Mcnemar's Test P-Value : 0.000437        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9806   0.9100   0.9527   0.9291   0.9518
## Specificity            0.9880   0.9883   0.9752   0.9931   0.9910
## Pos Pred Value         0.9702   0.9489   0.8903   0.9632   0.9598
## Neg Pred Value         0.9923   0.9786   0.9899   0.9862   0.9892
## Prevalence             0.2842   0.1935   0.1745   0.1638   0.1840
## Detection Rate         0.2787   0.1760   0.1663   0.1522   0.1751
## Detection Prevalence   0.2873   0.1855   0.1867   0.1580   0.1825
## Balanced Accuracy      0.9843   0.9491   0.9640   0.9611   0.9714
```

```r
# Algorithm 2 Boosting

confusionMatrix(pred2_B,validation2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 902  21   1   1   5
##          B  16 581  21   4   9
##          C   5  21 539  17  13
##          D   4   4   9 508   9
##          E   3   6   0   6 565
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9465          
##                  95% CI : (0.9382, 0.9539)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.9323          
##  Mcnemar's Test P-Value : 0.01363         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9699   0.9179   0.9456   0.9478   0.9401
## Specificity            0.9880   0.9810   0.9793   0.9905   0.9944
## Pos Pred Value         0.9699   0.9208   0.9059   0.9513   0.9741
## Neg Pred Value         0.9880   0.9803   0.9884   0.9898   0.9866
## Prevalence             0.2844   0.1936   0.1743   0.1639   0.1838
## Detection Rate         0.2758   0.1777   0.1648   0.1554   0.1728
## Detection Prevalence   0.2844   0.1930   0.1820   0.1633   0.1774
## Balanced Accuracy      0.9790   0.9494   0.9624   0.9691   0.9672
```

```r
# Algorithm 3 Boosting

confusionMatrix(pred3_B,validation3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 909  16   2   1   1
##          B   9 578  12   3  14
##          C   7  33 547  25   5
##          D   4   0   6 500   3
##          E   1   5   3   7 578
## 
## Overall Statistics
##                                          
##                Accuracy : 0.952          
##                  95% CI : (0.9441, 0.959)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9393         
##  Mcnemar's Test P-Value : 4.93e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9774   0.9146   0.9596   0.9328   0.9617
## Specificity            0.9914   0.9856   0.9741   0.9952   0.9940
## Pos Pred Value         0.9785   0.9383   0.8865   0.9747   0.9731
## Neg Pred Value         0.9910   0.9796   0.9913   0.9869   0.9914
## Prevalence             0.2845   0.1933   0.1744   0.1640   0.1838
## Detection Rate         0.2781   0.1768   0.1673   0.1530   0.1768
## Detection Prevalence   0.2842   0.1884   0.1887   0.1569   0.1817
## Balanced Accuracy      0.9844   0.9501   0.9669   0.9640   0.9779
```
In this case the average accuracy is 0.946 with the worst 95% confidence interval of (0.931,0.948).

For this particular datasets the Random Forest algorithms have shown better results in order to predict the classe of the dumbbell lifting. Among the three algorithms developed the one with the best results is the number two so it will be selected as our definitive algorithm.

As the algorithm has been developed splitting the training set in a new training set and a validation set, the results obtained when applying the algorithm to these validation sets should be very close to what we would obtain in an out sample of dataset. 

## Applying the random forest algorithm to the testing dataset


```r
Outsample_Prediction<-predict(modFit_RF_2,newdata=data.frame(testing_data_2))

Outsample_Prediction
```

```
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
```

## Appendix


```r
head(training_data)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         11      1.41       8.07    -94.4                3
## 2         no         11      1.41       8.07    -94.4                3
## 3         no         11      1.42       8.07    -94.4                3
## 4         no         12      1.48       8.05    -94.4                3
## 5         no         12      1.48       8.07    -94.4                3
## 6         no         12      1.45       8.06    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                                                         
## 2                                                         
## 3                                                         
## 4                                                         
## 5                                                         
## 6                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                                                                      NA
## 2                                                                      NA
## 3                                                                      NA
## 4                                                                      NA
## 5                                                                      NA
## 6                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1             NA                         NA             NA             
## 2             NA                         NA             NA             
## 3             NA                         NA             NA             
## 4             NA                         NA             NA             
## 5             NA                         NA             NA             
## 6             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                  NA                   NA                   
## 2                  NA                   NA                   
## 3                  NA                   NA                   
## 4                  NA                   NA                   
## 5                  NA                   NA                   
## 6                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                   NA            NA               NA            NA
## 2                   NA            NA               NA            NA
## 3                   NA            NA               NA            NA
## 4                   NA            NA               NA            NA
## 5                   NA            NA               NA            NA
## 6                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1             NA                NA             NA           NA
## 2             NA                NA             NA           NA
## 3             NA                NA             NA           NA
## 4             NA                NA             NA           NA
## 5             NA                NA             NA           NA
## 6             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1              NA           NA         0.00         0.00        -0.02
## 2              NA           NA         0.02         0.00        -0.02
## 3              NA           NA         0.00         0.00        -0.02
## 4              NA           NA         0.02         0.00        -0.03
## 5              NA           NA         0.02         0.02        -0.02
## 6              NA           NA         0.02         0.00        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1          -21            4           22            -3           599
## 2          -22            4           22            -7           608
## 3          -20            5           23            -2           600
## 4          -22            3           21            -6           604
## 5          -21            2           24            -6           600
## 6          -21            4           21             0           603
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1          -313     -128      22.5    -161              34            NA
## 2          -311     -128      22.5    -161              34            NA
## 3          -305     -128      22.5    -161              34            NA
## 4          -310     -128      22.1    -161              34            NA
## 5          -302     -128      22.1    -161              34            NA
## 6          -312     -128      22.0    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 1           NA              NA           NA            NA               NA
## 2           NA              NA           NA            NA               NA
## 3           NA              NA           NA            NA               NA
## 4           NA              NA           NA            NA               NA
## 5           NA              NA           NA            NA               NA
## 6           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 1            NA          NA             NA          NA        0.00
## 2            NA          NA             NA          NA        0.02
## 3            NA          NA             NA          NA        0.02
## 4            NA          NA             NA          NA        0.02
## 5            NA          NA             NA          NA        0.00
## 6            NA          NA             NA          NA        0.02
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 1        0.00       -0.02        -288         109        -123         -368
## 2       -0.02       -0.02        -290         110        -125         -369
## 3       -0.02       -0.02        -289         110        -126         -368
## 4       -0.03        0.02        -289         111        -123         -372
## 5       -0.03        0.00        -289         111        -123         -374
## 6       -0.03        0.00        -289         111        -122         -369
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 1          337          516                                     
## 2          337          513                                     
## 3          344          513                                     
## 4          344          512                                     
## 5          337          506                                     
## 6          342          513                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 1                                                                       
## 2                                                                       
## 3                                                                       
## 4                                                                       
## 5                                                                       
## 6                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 1           NA            NA          NA           NA            NA
## 2           NA            NA          NA           NA            NA
## 3           NA            NA          NA           NA            NA
## 4           NA            NA          NA           NA            NA
## 5           NA            NA          NA           NA            NA
## 6           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 1          NA                 NA                  NA                NA
## 2          NA                 NA                  NA                NA
## 3          NA                 NA                  NA                NA
## 4          NA                 NA                  NA                NA
## 5          NA                 NA                  NA                NA
## 6          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1      13.05217      -70.49400    -84.87394                       
## 2      13.13074      -70.63751    -84.71065                       
## 3      12.85075      -70.27812    -85.14078                       
## 4      13.43120      -70.39379    -84.87363                       
## 5      13.37872      -70.42856    -84.85306                       
## 6      13.38246      -70.81759    -84.46500                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 1                                                                     
## 2                                                                     
## 3                                                                     
## 4                                                                     
## 5                                                                     
## 6                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 1                                                              NA
## 2                                                              NA
## 3                                                              NA
## 4                                                              NA
## 5                                                              NA
## 6                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 1                 NA                                 NA                 NA
## 2                 NA                                 NA                 NA
## 3                 NA                                 NA                 NA
## 4                 NA                                 NA                 NA
## 5                 NA                                 NA                 NA
## 6                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 1                                       NA                       NA
## 2                                       NA                       NA
## 3                                       NA                       NA
## 4                                       NA                       NA
## 5                                       NA                       NA
## 6                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                                          37                 NA
## 2                                          37                 NA
## 3                                          37                 NA
## 4                                          37                 NA
## 5                                          37                 NA
## 6                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                NA                   NA                NA
## 2                NA                   NA                NA
## 3                NA                   NA                NA
## 4                NA                   NA                NA
## 5                NA                   NA                NA
## 6                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 1                 NA                    NA                 NA
## 2                 NA                    NA                 NA
## 3                 NA                    NA                 NA
## 4                 NA                    NA                 NA
## 5                 NA                    NA                 NA
## 6                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 1               NA                  NA               NA                0
## 2               NA                  NA               NA                0
## 3               NA                  NA               NA                0
## 4               NA                  NA               NA                0
## 5               NA                  NA               NA                0
## 6               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 1            -0.02             0.00             -234               47
## 2            -0.02             0.00             -233               47
## 3            -0.02             0.00             -232               46
## 4            -0.02            -0.02             -232               48
## 5            -0.02             0.00             -233               48
## 6            -0.02             0.00             -234               48
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 1             -271              -559               293               -65
## 2             -269              -555               296               -64
## 3             -270              -561               298               -63
## 4             -269              -552               303               -60
## 5             -270              -554               292               -68
## 6             -269              -558               294               -66
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 1         28.4         -63.9        -153                      
## 2         28.3         -63.9        -153                      
## 3         28.3         -63.9        -152                      
## 4         28.1         -63.9        -152                      
## 5         28.0         -63.9        -152                      
## 6         27.9         -63.9        -152                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 1                                                                  
## 2                                                                  
## 3                                                                  
## 4                                                                  
## 5                                                                  
## 6                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 1                                                           NA
## 2                                                           NA
## 3                                                           NA
## 4                                                           NA
## 5                                                           NA
## 6                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 1                NA                               NA                NA
## 2                NA                               NA                NA
## 3                NA                               NA                NA
## 4                NA                               NA                NA
## 5                NA                               NA                NA
## 6                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 1                                     NA                      NA
## 2                                     NA                      NA
## 3                                     NA                      NA
## 4                                     NA                      NA
## 5                                     NA                      NA
## 6                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 1                                        36                NA
## 2                                        36                NA
## 3                                        36                NA
## 4                                        36                NA
## 5                                        36                NA
## 6                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1               NA                  NA               NA                NA
## 2               NA                  NA               NA                NA
## 3               NA                  NA               NA                NA
## 4               NA                  NA               NA                NA
## 5               NA                  NA               NA                NA
## 6               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                   NA                NA              NA
## 2                   NA                NA              NA
## 3                   NA                NA              NA
## 4                   NA                NA              NA
## 5                   NA                NA              NA
## 6                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                 NA              NA            0.03            0.00
## 2                 NA              NA            0.02            0.00
## 3                 NA              NA            0.03           -0.02
## 4                 NA              NA            0.02           -0.02
## 5                 NA              NA            0.02            0.00
## 6                 NA              NA            0.02           -0.02
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.02             192             203            -215
## 2           -0.02             192             203            -216
## 3            0.00             196             204            -213
## 4            0.00             189             206            -214
## 5           -0.02             189             206            -214
## 6           -0.03             193             203            -215
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1              -17              654              476      A
## 2              -18              661              473      A
## 3              -18              658              469      A
## 4              -16              658              469      A
## 5              -17              655              473      A
## 6               -9              660              478      A
```

```r
head(testing_data)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1     pedro           1323095002               868349 05/12/2011 14:23
## 2 2    jeremy           1322673067               778725 30/11/2011 17:11
## 3 3    jeremy           1322673075               342967 30/11/2011 17:11
## 4 4    adelmo           1322832789               560311 02/12/2011 13:33
## 5 5    eurico           1322489635               814776 28/11/2011 14:13
## 6 6    jeremy           1322673149               510661 30/11/2011 17:12
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         74    123.00      27.00    -4.75               20
## 2         no        431      1.02       4.87   -88.90                4
## 3         no        439      0.87       1.82   -88.50                5
## 4         no        194    125.00     -41.60   162.00               17
## 5         no        235      1.35       3.33   -88.60                3
## 6         no        504     -5.92       1.59   -87.70                4
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                 NA                  NA                NA
## 2                 NA                  NA                NA
## 3                 NA                  NA                NA
## 4                 NA                  NA                NA
## 5                 NA                  NA                NA
## 6                 NA                  NA                NA
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                 NA                   NA                NA            NA
## 2                 NA                   NA                NA            NA
## 3                 NA                   NA                NA            NA
## 4                 NA                   NA                NA            NA
## 5                 NA                   NA                NA            NA
## 6                 NA                   NA                NA            NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1             NA           NA            NA             NA           NA
## 2             NA           NA            NA             NA           NA
## 3             NA           NA            NA             NA           NA
## 4             NA           NA            NA             NA           NA
## 5             NA           NA            NA             NA           NA
## 6             NA           NA            NA             NA           NA
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                  NA                   NA                 NA
## 2                  NA                   NA                 NA
## 3                  NA                   NA                 NA
## 4                  NA                   NA                 NA
## 5                  NA                   NA                 NA
## 6                  NA                   NA                 NA
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                   NA            NA               NA            NA
## 2                   NA            NA               NA            NA
## 3                   NA            NA               NA            NA
## 4                   NA            NA               NA            NA
## 5                   NA            NA               NA            NA
## 6                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1             NA                NA             NA           NA
## 2             NA                NA             NA           NA
## 3             NA                NA             NA           NA
## 4             NA                NA             NA           NA
## 5             NA                NA             NA           NA
## 6             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1              NA           NA        -0.50        -0.02        -0.46
## 2              NA           NA        -0.06        -0.02        -0.07
## 3              NA           NA         0.05         0.02         0.03
## 4              NA           NA         0.11         0.11        -0.16
## 5              NA           NA         0.03         0.02         0.00
## 6              NA           NA         0.10         0.05        -0.13
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1          -38           69         -179           -13           581
## 2          -13           11           39            43           636
## 3            1           -1           49            29           631
## 4           46           45         -156           169           608
## 5           -8            4           27            33           566
## 6          -11          -16           38            31           638
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1          -382     40.7    -27.80     178              10            NA
## 2          -309      0.0      0.00       0              38            NA
## 3          -312      0.0      0.00       0              44            NA
## 4          -304   -109.0     55.00    -142              25            NA
## 5          -418     76.1      2.76     102              29            NA
## 6          -291      0.0      0.00       0              14            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 1           NA              NA           NA            NA               NA
## 2           NA              NA           NA            NA               NA
## 3           NA              NA           NA            NA               NA
## 4           NA              NA           NA            NA               NA
## 5           NA              NA           NA            NA               NA
## 6           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 1            NA          NA             NA          NA       -1.65
## 2            NA          NA             NA          NA       -1.17
## 3            NA          NA             NA          NA        2.10
## 4            NA          NA             NA          NA        0.22
## 5            NA          NA             NA          NA       -1.96
## 6            NA          NA             NA          NA        0.02
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 1        0.48       -0.18          16          38          93         -326
## 2        0.85       -0.43        -290         215         -90         -325
## 3       -1.36        1.13        -341         245         -87         -264
## 4       -0.51        0.92        -238         -57           6         -173
## 5        0.79       -0.54        -197         200         -30         -170
## 6        0.05       -0.07         -26         130         -19          396
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 1          385          481                NA                 NA
## 2          447          434                NA                 NA
## 3          474          413                NA                 NA
## 4          257          633                NA                 NA
## 5          275          617                NA                 NA
## 6          176          516                NA                 NA
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 1               NA                NA                 NA               NA
## 2               NA                NA                 NA               NA
## 3               NA                NA                 NA               NA
## 4               NA                NA                 NA               NA
## 5               NA                NA                 NA               NA
## 6               NA                NA                 NA               NA
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 1           NA            NA          NA           NA            NA
## 2           NA            NA          NA           NA            NA
## 3           NA            NA          NA           NA            NA
## 4           NA            NA          NA           NA            NA
## 5           NA            NA          NA           NA            NA
## 6           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 1          NA                 NA                  NA                NA
## 2          NA                 NA                  NA                NA
## 3          NA                 NA                  NA                NA
## 4          NA                 NA                  NA                NA
## 5          NA                 NA                  NA                NA
## 6          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1     -17.73748       24.96085    126.23596                     NA
## 2      54.47761      -53.69758    -75.51480                     NA
## 3      57.07031      -51.37303    -75.20287                     NA
## 4      43.10927      -30.04885   -103.32003                     NA
## 5    -101.38396      -53.43952    -14.19542                     NA
## 6      62.18750      -50.55595    -71.12063                     NA
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 1                      NA                    NA                     NA
## 2                      NA                    NA                     NA
## 3                      NA                    NA                     NA
## 4                      NA                    NA                     NA
## 5                      NA                    NA                     NA
## 6                      NA                    NA                     NA
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 1                      NA                    NA                NA
## 2                      NA                    NA                NA
## 3                      NA                    NA                NA
## 4                      NA                    NA                NA
## 5                      NA                    NA                NA
## 6                      NA                    NA                NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 1                 NA               NA                NA                 NA
## 2                 NA               NA                NA                 NA
## 3                 NA               NA                NA                 NA
## 4                 NA               NA                NA                 NA
## 5                 NA               NA                NA                 NA
## 6                 NA               NA                NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 1               NA                      NA                       NA
## 2               NA                      NA                       NA
## 3               NA                      NA                       NA
## 4               NA                      NA                       NA
## 5               NA                      NA                       NA
## 6               NA                      NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                     NA                    9                 NA
## 2                     NA                   31                 NA
## 3                     NA                   29                 NA
## 4                     NA                   18                 NA
## 5                     NA                    4                 NA
## 6                     NA                   29                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                NA                   NA                NA
## 2                NA                   NA                NA
## 3                NA                   NA                NA
## 4                NA                   NA                NA
## 5                NA                   NA                NA
## 6                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 1                 NA                    NA                 NA
## 2                 NA                    NA                 NA
## 3                 NA                    NA                 NA
## 4                 NA                    NA                 NA
## 5                 NA                    NA                 NA
## 6                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 1               NA                  NA               NA             0.64
## 2               NA                  NA               NA             0.34
## 3               NA                  NA               NA             0.39
## 4               NA                  NA               NA             0.10
## 5               NA                  NA               NA             0.29
## 6               NA                  NA               NA            -0.59
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 1             0.06            -0.61               21              -15
## 2             0.05            -0.71             -153              155
## 3             0.14            -0.34             -141              155
## 4            -0.02             0.05              -51               72
## 5            -0.47            -0.46              -18              -30
## 6             0.80             1.10             -138              166
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 1               81               523              -528               -56
## 2             -205              -502               388               -36
## 3             -196              -506               349                41
## 4             -148              -576               238                53
## 5               -5              -424               252               312
## 6             -186              -543               262                96
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 1          141         49.30       156.0                    NA
## 2          109        -17.60       106.0                    NA
## 3          131        -32.60        93.0                    NA
## 4            0          0.00         0.0                    NA
## 5         -176         -2.16       -47.9                    NA
## 6          150          1.46        89.7                    NA
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 1                     NA                   NA                    NA
## 2                     NA                   NA                    NA
## 3                     NA                   NA                    NA
## 4                     NA                   NA                    NA
## 5                     NA                   NA                    NA
## 6                     NA                   NA                    NA
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 1                     NA                   NA               NA
## 2                     NA                   NA               NA
## 3                     NA                   NA               NA
## 4                     NA                   NA               NA
## 5                     NA                   NA               NA
## 6                     NA                   NA               NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 1                NA              NA               NA                NA
## 2                NA              NA               NA                NA
## 3                NA              NA               NA                NA
## 4                NA              NA               NA                NA
## 5                NA              NA               NA                NA
## 6                NA              NA               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 1              NA                     NA                      NA
## 2              NA                     NA                      NA
## 3              NA                     NA                      NA
## 4              NA                     NA                      NA
## 5              NA                     NA                      NA
## 6              NA                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 1                    NA                  33                NA
## 2                    NA                  39                NA
## 3                    NA                  34                NA
## 4                    NA                  43                NA
## 5                    NA                  24                NA
## 6                    NA                  43                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1               NA                  NA               NA                NA
## 2               NA                  NA               NA                NA
## 3               NA                  NA               NA                NA
## 4               NA                  NA               NA                NA
## 5               NA                  NA               NA                NA
## 6               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                   NA                NA              NA
## 2                   NA                NA              NA
## 3                   NA                NA              NA
## 4                   NA                NA              NA
## 5                   NA                NA              NA
## 6                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                 NA              NA            0.74           -3.34
## 2                 NA              NA            1.12           -2.78
## 3                 NA              NA            0.18           -0.79
## 4                 NA              NA            1.38            0.69
## 5                 NA              NA           -0.75            3.10
## 6                 NA              NA           -0.88            4.26
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.59            -110             267            -149
## 2           -0.18             212             297            -118
## 3            0.28             154             271            -129
## 4            1.80             -92             406             -39
## 5            0.80             131             -93             172
## 6            1.35             230             322            -144
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z problem_id
## 1             -714              419              617          1
## 2             -237              791              873          2
## 3              -51              698              783          3
## 4             -233              783              521          4
## 5              375             -787               91          5
## 6             -300              800              884          6
```

```r
names(training_data)==names(testing_data)
```

```
##   [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [12]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [23]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [34]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [45]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [56]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [67]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [78]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [89]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [100]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [111]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [122]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [133]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [144]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [155]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
```
