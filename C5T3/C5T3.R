# Parallel Processing
# Install packages and import libraries
install.packages("doParallel")
library(doParallel) # For parallel processing
library(plotly) # For graphing
library(corrplot) # Correlation graph
library(caret) # Machine learning
library(mlbench)
library(e1071) # SVM 

# Find how many cores are on machine
detectCores() # Result = 4

# Create Cluster with 2 cores 
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result = 2 

# Stop Cluster
stopCluster(cl)

#########################################################
# Explore Data
#########################################################

# Take a look at dtype (int) and summary of data
str(iphone)
summary(iphone)

# Check for null values
sum(is.na(iphone))

# View a histogram of iphonesentiment (dependent var)
plot_ly(iphone, x = iphone$iphonesentiment, type = 'histogram')

# Change maxprint options
options(max.print = 1000000)

###########################################################
# View correlation amongst features
x = cor(iphone)
corrplot(x, tl.cex = 0.5) # tl.cex changes text size of graph
print(x)

# Features tend to correlate negatively with iphonesentiment
# Features that are near zero correlation with the dependent all 
# tend to be about the iphone so I won't be removing these features

# Find highly correlated features (with abs corr of 0.75 or higher)
highCorr <- findCorrelation(x, cutoff = 0.75, names = TRUE)
print(highCorr)

## To remove values from dataset
## highCorrIndex <- findCorrelation(x, cutoff = 0.75)
## highCorrIndex <- sort(highCorrIndex)
## iphoneCOR <- iphone[,-c(highCorrIndex)]

## Will not be removing these features as this does not really
## apply to a classification problem.


###########################################################
# Near-zero variance
nzvMetrics <- nearZeroVar(iphone, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(iphone, saveMetrics = FALSE) 
nzv

# Create a new dataset with the nzv features removed
iphoneNZV <- iphone[,-nzv]
str(iphoneNZV)

############################################################
# Recursive Feature Elimination (RFE)
# First create a sample of the data
set.seed(123)
iphoneSample <- iphone[sample(1:nrow(iphone), 1000, replace = FALSE),]

# Set up rfeControl with randomforest, repeated cv, and no updates
ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv",
                   repeats = 5, verbose = FALSE)

# Use RFE and omit response var (attribute 59)
rfeResults <- rfe(iphoneSample[,1:58], iphoneSample$iphonesentiment, 
                  sizes = (1:58), rfeControl = ctrl)
rfeResults

## The top 5 variables (out of 19):
## iphone, googleandroid, iphonedispos, iphonedisneg, samsunggalaxy

# Plot the RFE results
plot(rfeResults, type = c("g", "o"))

# Create new data set with rfe recommended features
iphoneRFE <- iphone[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone$iphonesentiment

# review outcome
str(iphoneRFE)

###################################
# Change iphonesentiment to a factor for each dataset

iphone$iphonesentiment <- as.factor(iphone$iphonesentiment)
str(iphone$iphonesentiment)
iphoneNZV$iphonesentiment <- as.factor(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)

##############################################################
## Model Development and Evaluation
##############################################################

## Four different models will be trained on each data set

##############
## OUT OF BOX DATASET
## C50
set.seed(123)
# Create Training and Test sets with 70/30 split
inTraining <- createDataPartition(iphone$iphonesentiment, p = .70, list = FALSE)
training <- iphone[inTraining,]
testing <- iphone[-inTraining,]

# Establish 10-fold repeated cross-validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# Train C50 model
c50Fit1 <- train(iphonesentiment~., data = training, method = "C5.0", 
                 trControl = fitControl)
# View model results
c50Fit1

# model  winnow  trials  Accuracy   Kappa  
# tree   FALSE    1      0.7718829  0.5577752

# Make predictions on test set
c50Pred1 <- predict(c50Fit1, testing)
c50Pred1

# How well do predictions align with ground truth
postResample(c50Pred1, testing$iphonesentiment)

# Accuracy     Kappa 
# 0.7712082    0.5542280 

# Confusion Matrix
cmC501 <- confusionMatrix(c50Pred1, testing$iphonesentiment)
cmC501 # model was better at predicting 0, 4, 5, but horrible at other values
# This is likely bc of the distribution of the ratings -- there are many more 5's
# than other ratings, so the model tends to skew more towards predicting 5's, 
# whereas there are far fewer 1's, so the model does not do well with those

##############
# OUT OF BOX DATASET
# RANDOM FOREST
# Train model
set.seed(123)
rfFit1 <- train(iphonesentiment~., data = training, method = "rf",
                trControl = fitControl)
# View results
rfFit1

# mtry  Accuracy   Kappa 
# 30    0.7728743  0.5629760

# Make predictions
rfPred1 <- predict(rfFit1, testing)
rfPred1

# Post Resample
postResample(rfPred1, testing$iphonesentiment)

# Accuracy     Kappa 
# 0.7732648    0.5603207

# Confusion Matrix
cmRF1 <- confusionMatrix(rfPred1, testing$iphonesentiment)
cmRF1 # Results are similar to c50 model

##############
# OUT OF BOX DATASET
# SVM LINEAR
# Train model
set.seed(123)
svmFit1 <- train(iphonesentiment~., data = training, method = "svmLinear2",
                 trControl = fitControl)
# View Results
svmFit1

# cost  Accuracy   Kappa
# 0.50  0.7092393  0.4114014

# Make predictions
svmPred1 <- predict(svmFit1, testing)
svmPred1

# Post Resample
postResample(svmPred1, testing$iphonesentiment)

# Accuracy     Kappa 
# 0.7120823 0.4200502 

# Confusion Matrix
cmSVM1 <- confusionMatrix(svmPred1, testing$iphonesentiment)
cmSVM1 # Similar issue as above

##############
# OUT OF BOX DATASET
# KKNN
# Train model
set.seed(123)
kknnFit1 <- train(iphonesentiment~., data = training, method = "kknn",
                  trControl = fitControl)
# View Results
kknnFit1

# kmax  Accuracy   Kappa 
# 9     0.3283062  0.1609948

# Not moving forward w this model

#############################################
#############
# NZV DATASET
# C50
# Create Training and Test set for iphoneNZV data
set.seed(123)
inTrainingNZV <- createDataPartition(iphoneNZV$iphonesentiment, p = .70, list = FALSE)
trainingNZV <- iphoneNZV[inTrainingNZV,]
testingNZV <- iphoneNZV[-inTrainingNZV,]

# Train C50 model
c50FitNZV <- train(iphonesentiment~., data = trainingNZV, method = "C5.0", 
                 trControl = fitControl)
# View Results
c50FitNZV

# model  winnow  trials  Accuracy   Kappa 
# tree   FALSE    1      0.7556945  0.5204503

# Make predictions
c50PredNZV <- predict(c50FitNZV, testing)

# Post Resample
postResample(c50PredNZV, testingNZV$iphonesentiment)

# Accuracy     Kappa 
# 0.7555270    0.5196516

# Confusion Matrix
cmC50NZV <- confusionMatrix(c50PredNZV, testingNZV$iphonesentiment)
cmC50NZV

#############
# NZV DATASET
# RANDOM FOREST
# Train model
set.seed(123)
rfFitNZV <- train(iphonesentiment~., data = trainingNZV, method = "rf", 
                   trControl = fitControl)
# View Results
rfFitNZV

#   mtry  Accuracy   Kappa    
#    2    0.7602141  0.5281270

# Make Predictions
rfPredNZV <- predict(rfFitNZV, testingNZV)

# Post Resample
postResample(rfPredNZV, testingNZV$iphonesentiment)

# Confusion Matrix
cmRfNZV <- confusionMatrix(rfPredNZV, testingNZV$iphonesentiment)
cmRfNZV # Models are all having similar issue... algo is better at classifying 5's

#############
# NZV DATASET
# SVM LINEAR
# Train model
set.seed(123)
svmFitNZV <- train(iphonesentiment~., data = trainingNZV, method = "svmLinear2",
                   trControl = fitControl)
# View Results
svmFitNZV

# cost  Accuracy   Kappa
# 1.00  0.6832542  0.3450395

# Not to continue with model

############
# NZV DATASET
# KKNN
# Train model
set.seed(123)
kknnFitNZV <- train(iphonesentiment~., data = trainingNZV, method = "kknn",
                    trControl = fitControl)
# View Results
kknnFitNZV

# kmax  Accuracy   Kappa 
# 9     0.3087106  0.1335733

# Not to continue with model

###################################################
##############
# RFE DATASET
# C50
# Create training and test set for iphoneRFE data
set.seed(123)
inTrainingRFE <- createDataPartition(iphoneRFE$iphonesentiment, p = .70, list = FALSE)
trainingRFE <- iphoneRFE[inTrainingRFE,]
testingRFE <- iphoneRFE[-inTrainingRFE,]

# Train model
c50FitRFE <- train(iphonesentiment~., data = trainingRFE, method = "C5.0",
                   trControl = fitControl)
# View results
c50FitRFE

# model  winnow  trials  Accuracy   Kappa
# rules   TRUE    1      0.7727593  0.5584020

# Make predictions
c50PredRFE <- predict(c50FitRFE, testingRFE)

# Post Resample
postResample(c50PredRFE, testingRFE$iphonesentiment)

# Accuracy     Kappa 
# 0.7701799    0.5520520

# Confusion Matrix
cmC50RFE <- confusionMatrix(c50PredRFE, testingRFE$iphonesentiment)
cmC50RFE

#############
# RFE DATASET
# RANDOM FOREST
# Train model
rfFitRFE <- train(iphonesentiment~., data = trainingRFE, method = "rf",
                   trControl = fitControl)
# View Results
rfFitRFE

# mtry  Accuracy   Kappa
# 10    0.7735343  0.5646091

# Make Predictions
rfPredRFE <- predict(rfFitRFE, testingRFE)

# Post Resample
postResample(rfPredRFE, testingRFE$iphonesentiment)

# Accuracy     Kappa 
# 0.7722365    0.5589829

# Confusion Matrix
cmRfRFE <- confusionMatrix(rfPredRFE, testingRFE$iphonesentiment)
cmRfRFE

#############
# RFE DATASET
# SVM LINEAR
# Train model
svmFitRFE <- train(iphonesentiment~., data = trainingRFE, method = "svmLinear2",
                   trControl = fitControl)
# View Results
svmFitRFE

# cost  Accuracy   Kappa 
# 1.00  0.7104506  0.4174320

# Not to continue with model

############
# RFE DATASET
# KKNN
# Train model
kknnFitRFE <- train(iphonesentiment~., data = trainingRFE, method = "kknn",
                    trControl = fitControl)
# View results
kknnFitRFE

# kmax  Accuracy   Kappa
# 9     0.3334716  0.1662437

# Not to continue with model

###########################################################
# FEATURE ENGINEERING
###########################################################

# Change the dependent variable. All models struggled with sensitivity/
# specificity of some of the rating numbers 
# I will combine some of them to condense the dependent variable and give it
# a new rating system.
# 1: negative, 2: somewhat negative, 3: somewhat positive, 4: positive

library(dplyr) # for recode()
# Create a new dataset that will be used for recoding sentiment
iphoneRC <- iphone

# Recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, 
                                   '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# Inspect results
summary(iphoneRC)
str(iphoneRC)

# Look at hist of iphonesentiment
plot_ly(iphoneRC, x = iphoneRC$iphonesentiment, type = 'histogram')

# Make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

########################
# Test C50 with new dependent variable

set.seed(123)
# Create Training and Test sets with 70/30 split
inTrainingRC <- createDataPartition(iphoneRC$iphonesentiment, p = .70, list = FALSE)
trainingRC <- iphoneRC[inTrainingRC,]
testingRC <- iphoneRC[-inTrainingRC,]

# Train model
set.seed(123)
c50FitRC <- train(iphonesentiment~., data = trainingRC, method = "C5.0",
                trControl = fitControl)
# View results
c50FitRC

# C5.0 results
# model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.8471884  0.6172888

# Random Forest Results -- did not choose this model due to long modeling time
# but not significant improvement in results
# mtry  Accuracy   Kappa 
# 30    0.8477391  0.6223489
# Results improved significantly

# Make predictions
rfPredRC <- predict(rfFitRC, testingRC)

# Post Resample
postResample(rfPredRC, testingRC$iphonesentiment)

# Accuracy     Kappa 
# 0.8447301 0.6123915  

# Confusion Matrix
cmRFRC <- confusionMatrix(rfPredRC, testingRC$iphonesentiment)
cmRFRC 

#                       Class: 1 Class: 2 Class: 3 Class: 4
# Sensitivity           0.54610 0.088235  0.62921   0.9896
# Specificity           0.99309 1.000000  0.99491   0.5288

# Results are better when combining classes, however, still 
# horrible for Class 2


###########################
# Principal Component Analysis

# Data = OUT OF BOX dataset
# Create object containing centered, scaled PCA components from training set
# Excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# Use predict to apply PCA parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, training[,-59])

# Add the dependent to training
train.pca$iphonesentiment <- training$iphonesentiment

# Use predict to apply PCA parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testing[,-59])

# Add the dependent to training
test.pca$iphonesentiment <- testing$iphonesentiment

# Inspect results
str(train.pca)
str(test.pca)

# Train model
set.seed(123)
rfFitPCA <- train(iphonesentiment~., data = train.pca, method = "rf",
                  trControl = fitControl)
rfFitPCA
# mtry  Accuracy   Kappa
# 13    0.7606472  0.5414521

# Make predictions
rfPredPCA <- predict(rfFitPCA, test.pca)

# Post Resample
postResample(rfPredPCA, test.pca$iphonesentiment)

# Accuracy     Kappa 
# 0.7642674 0.5446032

# Confusion Matrix
cmRfPCA <- confusionMatrix(rfPredPCA, test.pca$iphonesentiment)
cmRfPCA

# Not much improvement

###############################################################
# Make Predictions on LargeMatrix Dataset
###############################################################

# First drop the 'id' column
iphoneLargeMatrix <- subset(iphoneLargeMatrix, select = -id)

# View str of dataset
str(iphoneLargeMatrix)
summary(iphoneLargeMatrix) #NA values where I will be making predictions

LGpred <- predict(c50FitRC, iphoneLargeMatrix)
LGpred
 
# Add predictions to the dataframe
iphoneLargeMatrix$iphonesentiment <- predict(c50FitRC, iphoneLargeMatrix)

# View histogram of results
plot_ly(iphoneLargeMatrix, x = iphoneLargeMatrix$iphonesentiment, type = 'histogram')
table(iphoneLargeMatrix$iphonesentiment)

#     1     2     3     4 
# 17378  2287  1329  9493 

# Create plot of sentiment for iphone
# create df
isentiment_level <- c('Negative', 'Somewhat Negative', 'Somewhat Positive', 'Positive')
icounts <- c(17378, 2287, 1329, 9493)
iplot <- data.frame(isentiment_level, icounts)
# Make sentiment_level a factor so chart will display in order
iplot$isentiment_level <- factor(iplot$isentiment_level, 
                                  levels = c("Negative", "Somewhat Negative", "Somewhat Positive", "Positive"))

# Create bar plot, changing color of bars
fig1 <- plot_ly(iplot, x = ~isentiment_level, y = ~icounts, type = 'bar',
                text = icounts, textposition = 'auto', 
                marker = list(color = c('#FF4500', 'pink',
                                        'lightblue', '#4169E1')))
# Create titles
fig1 <- fig1 %>% layout(title = "iPhone Sentiment",
                        xaxis = list(title = "Sentiment"),
                        yaxis = list(title = " "))
# View plot
fig1

###############################################################
###############################################################
## GALAXY DATASET
###############################################################
###############################################################

# Take a look at dtype (int) and summary of data
str(galaxy)
summary(galaxy)

# Check for null values
sum(is.na(galaxy))

# View a histogram of galaxysentiment (dependent var)
plot_ly(galaxy, x = galaxy$galaxysentiment, type = 'histogram')
# Very similar distribution to iphone

# Correlation
y = cor(galaxy)
corrplot(y, tl.cex = 0.5) # tl.cex changes text size of graph
print(y)

###############################################################
# NZV

nzvMetricsGal <- nearZeroVar(galaxy, saveMetrics = TRUE)
nzvMetricsGal

nzvGal <- nearZeroVar(galaxy, saveMetrics = FALSE) 
nzvGal

# Create a new dataset with the nzv features removed
galaxyNZV <- galaxy[,-nzv]
str(galaxyNZV)

############################################################
# Recursive Feature Elimination (RFE)
# First create a sample of the data
set.seed(123)
galaxySample <- galaxy[sample(1:nrow(galaxy), 1000, replace = FALSE),]

# Set up rfeControl with randomforest, repeated cv, and no updates
ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv",
                   repeats = 5, verbose = FALSE)

# Use RFE and omit response var (attribute 59)
rfeResultsGal <- rfe(galaxySample[,1:58], galaxySample$galaxysentiment, 
                  sizes = (1:58), rfeControl = ctrl)
rfeResultsGal

# The top 5 variables (out of 56):
# iphone, googleandroid, samsunggalaxy, iphoneperunc, iphoneperpos

# Plot the RFE results
plot(rfeResultsGal, type = c("g", "o"))

# Create new data set with rfe recommended features
galaxyRFE <- galaxy[,predictors(rfeResultsGal)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxy$galaxysentiment

# review outcome
str(galaxyRFE)

###################################
# Change galaxysentiment to a factor for each dataset

galaxy$galaxysentiment <- as.factor(galaxy$galaxysentiment)
str(galaxy$galaxysentiment)
galaxyNZV$galaxysentiment <- as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)

###############################################################
# Modeling and Evaluation
###############################################################

## OUT OF BOX DATASET
## C50
set.seed(123)
# Create Training and Test sets with 70/30 split
inTrainingGal <- createDataPartition(galaxy$galaxysentiment, p = .70, list = FALSE)
trainingGal <- galaxy[inTrainingGal,]
testingGal <- galaxy[-inTrainingGal,]

# Establish 10-fold repeated cross-validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# Train C50 model
c50FitG <- train(galaxysentiment~., data = trainingGal, method = "C5.0", 
                 trControl = fitControl)
# View model results
c50FitG

# model  winnow  trials  Accuracy   Kappa  
# rules  FALSE    1      0.7669282  0.5317926

# Make predictions on test set
c50PredG <- predict(c50FitG, testingGal)
c50PredG

# How well do predictions align with ground truth
postResample(c50PredG, testingGal$galaxysentiment)

# Accuracy     Kappa 
# 0.7675019 0.5322522 

# Confusion Matrix
cmC50G <- confusionMatrix(c50PredG, testingGal$galaxysentiment)
cmC50G 

##############
# OUT OF BOX DATASET
# RANDOM FOREST
# Train model
set.seed(123)
rfFitG <- train(galaxysentiment~., data = trainingGal, method = "rf",
                trControl = fitControl)
# View results
rfFitG

# RF took far too long to run so this model will be abandoned

##############
# OUT OF BOX DATASET
# SVM LINEAR
# Train model
set.seed(123)
svmFitG <- train(galaxysentiment~., data = trainingGal, method = "svmLinear2",
                 trControl = fitControl)
# View Results
svmFitG

# Model took too long to run

##############
# OUT OF BOX DATASET
# KKNN
# Train model
set.seed(123)
kknnFitG <- train(galaxysentiment~., data = trainingGal, method = "kknn",
                  trControl = fitControl)
# View Results
kknnFitG

# kmax  Accuracy   Kappa 
# 7     0.7336326  0.4887705

# Make predictions on test set
kknnPredG <- predict(kknnFitG, testingGal)
kknnPredG

# How well do predictions align with ground truth
postResample(kknnPredG, testingGal$galaxysentiment)

# Accuracy     Kappa 
# 0.7318522    0.4877367 

# Confusion Matrix
cmKKNNG <- confusionMatrix(kknnPredG, testingGal$galaxysentiment)
cmKKNNG 

#############################################
#############
# NZV DATASET
# C50
# Create Training and Test set for iphoneNZV data
set.seed(123)
inTrainingNZVG <- createDataPartition(galaxyNZV$galaxysentiment, p = .70, list = FALSE)
trainingNZVG <- galaxyNZV[inTrainingNZVG,]
testingNZVG <- galaxyNZV[-inTrainingNZVG,]

# Train C50 model
c50FitNZVG <- train(galaxysentiment~., data = trainingNZVG, method = "C5.0", 
                   trControl = fitControl)
# View Results
c50FitNZVG

# model  winnow  trials  Accuracy   Kappa 
# rules  FALSE    1      0.7538730  0.5006638

# Make predictions
c50PredNZVG <- predict(c50FitNZVG, testingNZVG)

# Post Resample
postResample(c50PredNZVG, testingNZVG$galaxysentiment)

# Accuracy     Kappa 
# 0.7514854 0.4936633 

# Confusion Matrix
cmC50NZVG <- confusionMatrix(c50PredNZVG, testingNZVG$galaxysentiment)
cmC50NZVG

#############
# NZV DATASET
# RANDOM FOREST
# Train model
set.seed(123)
rfFitNZVG <- train(galaxysentiment~., data = trainingNZVG, method = "rf", 
                  trControl = fitControl)
# View Results
rfFitNZVG

#   mtry  Accuracy   Kappa    
#   

# Make Predictions
rfPredNZVG <- predict(rfFitNZVG, testingNZVG)

# Post Resample
postResample(rfPredNZVG, testingNZVG$galaxysentiment)

# Confusion Matrix
cmRfNZVG <- confusionMatrix(rfPredNZVG, testingNZVG$galaxysentiment)
cmRfNZVG

#############
# NZV DATASET
# SVM LINEAR
# Train model
set.seed(123)
svmFitNZVG <- train(galaxysentiment~., data = trainingNZVG, method = "svmLinear2",
                   trControl = fitControl)
# View Results
svmFitNZVG

# cost  Accuracy   Kappa
# 

# Not to continue with model

############
# NZV DATASET
# KKNN
# Train model
set.seed(123)
kknnFitNZVG <- train(galaxysentiment~., data = trainingNZVG, method = "kknn",
                    trControl = fitControl)
# View Results
kknnFitNZVG

# kmax  Accuracy   Kappa 
# 7     0.7044408  0.4406507

###################################################
##############
# RFE DATASET
# C50
# Create training and test set for iphoneRFE data
set.seed(123)
inTrainingRFEG <- createDataPartition(galaxyRFE$galaxysentiment, p = .70, list = FALSE)
trainingRFEG <- galaxyRFE[inTrainingRFEG,]
testingRFEG <- galaxyRFE[-inTrainingRFEG,]

# Train model
c50FitRFEG <- train(galaxysentiment~., data = trainingRFEG, method = "C5.0",
                   trControl = fitControl)
# View results
c50FitRFEG

# model  winnow  trials  Accuracy   Kappa
# 

# Make predictions
c50PredRFEG <- predict(c50FitRFEG, testingRFEG)

# Post Resample
postResample(c50PredRFEG, testingRFEG$galaxysentiment)

# Accuracy     Kappa 
# 

# Confusion Matrix
cmC50RFEG <- confusionMatrix(c50PredRFEG, testingRFEG$galaxysentiment)
cmC50RFEG

##############################################################
##Feature Engineering
##############################################################

# Change the dependent variable
# Combine some of the ratings to make a new system
# 1: negative, 2: somewhat negative, 3: somewhat positive, 4: positive

library(dplyr) # for recode()
# Create a new dataset that will be used for recoding sentiment
galaxyRC <- galaxy

# Recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, 
                                   '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# Inspect results
summary(galaxyRC)
str(galaxyRC)

# Look at hist of iphonesentiment
plot_ly(galaxyRC, x = galaxyRC$galaxysentiment, type = 'histogram')

# Make iphonesentiment a factor
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)

# Test C50 with new dependent variable

set.seed(123)
# Create Training and Test sets with 70/30 split
inTrainingRCG <- createDataPartition(galaxyRC$galaxysentiment, p = .70, list = FALSE)
trainingRCG <- galaxyRC[inTrainingRCG,]
testingRCG <- galaxyRC[-inTrainingRCG,]

# Train model
set.seed(123)
c50FitRCG <- train(galaxysentiment~., data = trainingRCG, method = "C5.0",
                  trControl = fitControl)
# View results
c50FitRCG

# C5.0 results
# model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.8425723  0.5900179

set.seed(123)
kknnFitRCG <- train(galaxysentiment~., data = trainingRCG, method = "kknn",
                       trControl = fitControl)
kknnFitRCG

# kmax  # Accuracy  # Kappa
# 9     0.8322796  0.5693936

################################################################

# First drop the 'id' column
galaxyLargeMatrix <- subset(galaxyLargeMatrix, select = -id)

# View str of dataset
str(galaxyLargeMatrix)
summary(galaxyLargeMatrix)

LGgalpred <- predict(c50FitRCG, galaxyLargeMatrix)
LGgalpred

# Add predictions to the dataframe
galaxyLargeMatrix$galaxysentiment <- predict(c50FitRCG, galaxyLargeMatrix)

# View basic histogram of results
galfig <- plot_ly(galaxyLargeMatrix, x = galaxyLargeMatrix$galaxysentiment, 
                  type = 'histogram')
  
# View table of counts
table(galaxyLargeMatrix$galaxysentiment)

#     1     2     3     4 
#   18355  2081   784  9267 

# Create bar chart of results for galaxysentiment
# create df
sentiment_level <- c('Negative', 'Somewhat Negative', 'Somewhat Positive', 'Positive')
counts <- c(18355, 2081, 784, 9267)
galplot <- data.frame(sentiment_level, counts)
# Make sentiment_level a factor so chart will display in order
galplot$sentiment_level <- factor(galplot$sentiment_level, 
                                  levels = c("Negative", "Somewhat Negative", "Somewhat Positive", "Positive"))

# Create bar plot, changing color of bars
fig2 <- plot_ly(galplot, x = ~sentiment_level, y = ~counts, type = 'bar',
                text = counts, textposition = 'auto', 
                marker = list(color = c('#FF4500', 'pink',
                                'lightblue', '#4169E1')))
# Create titles
fig2 <- fig2 %>% layout(title = "Samsung Galaxy Sentiment",
                       xaxis = list(title = "Sentiment"),
                       yaxis = list(title = " "))
# View plot
fig2
