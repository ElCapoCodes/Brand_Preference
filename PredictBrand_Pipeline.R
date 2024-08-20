#! /usr/local/bin/Rscript

# Title: Predict Brand Preference

# Last update: 2021.10


###############
# Project Notes
###############

# Our task is to investigate and create a highly accurate model using survey information to predict
# which computer brand customers would prefer. This model would be an extra tool for our client’s
# sales force to understand customer’s preferences in computer choices.

###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
# set working directory
path <- file.path("/Home/BrandPref")
setwd(path)
dir()

################
# Load packages
################
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doMC)    # for OSX parallel processing 
#
# GRAPH Libraries:
#
library(lattice) 
#
library(dplyr)
library(explore)
library(data.table)
#
library(inum)
library(C50)
#
#####################
# Parallel processing
#####################
# NOTE: Be sure to use the correct package for your operating system. 
#--- for OSX ---#
detectCores()            # detect number of cores
registerDoMC(cores = 1)  # set number of cores (don't use all available)

##############
# Import data 
##############

##-- Load Train/Existing data (Dataset 1) --##
subdir <- paste0(getwd(),'/SurveyData')
wyCompOOB <- read.csv(paste0(subdir,'/CompleteResponses.csv'), stringsAsFactors = FALSE)
str(wyCompOOB)

##--- Load Predict/New data (Dataset 2) [dv = NA, 0, or Blank] ---##
wyIncompOOB <- read.csv(paste0(subdir,'/SurveyIncomplete.csv'), stringsAsFactors = FALSE)
str(wyIncompOOB)


###############
# Save datasets
###############

# can save all datasets to csv format, or 
# can save the ds with the best performance (after all modeling)
# write.csv(ds_object, "ds_name.csv", row.names = F)

################
# Evaluate data
################

##--- Dataset 1 ---##
##-------------------
str(wyCompOOB)  
# check for missing values 
anyNA(wyCompOOB)
# check for duplicates
anyDuplicated((wyCompOOB))

##--- Dataset 2 ---##
##-------------------
# NOTE: often don't have a Dataset 2 (dataset with dv = 0, NA, or Blank)
str(wyIncompOOB)
# check for missing values 
anyNA(wyIncompOOB) 
# check for duplicates
anyDuplicated(wyIncompOOB) 

#############
# Preprocess
#############

##--- Dataset 1 ---##

# remove ID and obvious features 
names(wyCompOOB)
str(wyCompOOB)
#wyCompOOB$feature_to_remove <- NULL   # remove ID, if applicable
#str(wyCompOOB) # confirm removed features
# rename a column A/= no need. Column names are short, lower case and logical
#names(ds) <- c("ColumnName","ColumnName","ColumnName") 

# change data types
# Need to change brand (Y-value) to factor so classification can run.

wyCompCor<- copy(wyCompOOB)

wyCompOOB$brand   <- as.factor(wyCompOOB$brand)
wyCompOOB$car     <- as.factor(wyCompOOB$car)
wyCompOOB$zipcode <- as.factor(wyCompOOB$zipcode)
wyCompOOB$elevel  <- as.factor(wyCompOOB$elevel)
str(wyCompOOB)
#
# handle missing values (if applicable)
#na.omit(ds$ColumnName)
#na.exclude(ds$ColumnName)        
#ds$ColumnName[is.na(ds$ColumnName)] <- mean(ds$ColumnName,na.rm = TRUE)


##--- Dataset 2 ---##
# If there is a dataset with unseen data to make predictions on, then preprocess 
# here to make sure that it is preprossed the same as the dataset that had
# the best results - e.g., oob or a tuned ds from feature selection/engineering.
# check for missing values 
# anyNA(wyIncompOOB)
wyIncompOOB$brand   <- as.factor(wyIncompOOB$brand)
wyIncompOOB$car     <- as.factor(wyIncompOOB$car)
wyIncompOOB$zipcode <- as.factor(wyIncompOOB$zipcode)
wyIncompOOB$elevel  <- as.factor(wyIncompOOB$elevel)
str(wyIncompOOB)
# do we expect the dv to have any blank/na values?
# evaluate if any NA in just the IV variables using indexing
# your code here

#####################
# EDA/Visualizations
#####################
names(wyCompOOB)
#
str(wyCompOOB) 
describe(wyCompOOB) #dlookr
#normality(wyCompOOB) # error dlookr
#
# statistics
summary(wyCompOOB)

# ---------------
# -- plots Library:
# ---------------
hist(wyCompOOB$age) 
#plot(wyCompOOB$car)
# pairs(wyCompOOB) 

# ---------------
# -- Lattice plots
# ---------------
#
histogram(~brand, data=wyCompOOB,type="percent") 
#
#
histogram(~elevel, data=wyCompOOB,type="percent") 
histogram(~car, data=wyCompOOB,type="percent") 
histogram(~age, data=wyCompOOB,type="percent") 
xyplot(salary~brand, data=wyCompOOB)
xyplot(salary~age, data=wyCompOOB) 
xyplot(zipcode~brand, data=wyCompOOB)

#######################
# Correlation analysis
#######################

correlations <- cor(wyCompCor)
dim(correlations) 
#
round(correlations, digits = 3) 
#
library(corrplot)
corrplot(correlations, order = "hclust") 

#######################
# Feature selection
#######################
#
# I: Not needed since we do not have highly correlated features.

# ------------------------------------------
# Recursive Feature Elimination (RFE)
# ------------------------------------------
# 
## ---- rf ---- ##

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)
#
# run the RFE algorithm
set.seed(7)
rfeRF <- rfe(wyCompOOB[,1:6], wyCompOOB[,7], sizes=c(1:6), rfeControl=RFcontrol)
rfeRF 
# plot the results
plot(rfeRF, type=c("g", "o"))
# show predictors used (car is dropped)
predictors(rfeRF)
# Note results.  
varImp(rfeRF)

##--- create ds with features using varImp from top model ---##

# create ds with predictors from varImp(RF) since we're doing classification
wyCompRFE <- wyCompOOB[,predictors(rfeRF)]
str(wyCompRFE)
# add dv
wyCompRFE$brand <- wyCompOOB$brand
# confirm new ds
str(wyCompRFE)

# ==
# ==================================================
# ==================================================
# ==      PREDICTIVE MODELS FOR wyCompOOB DATASET ==
# ==================================================
# ==================================================
# ==
#
# ---------------------------------
# -- 1. CREATE TRAIN/TEST SETS
# ---------------------------------
#
# I: createDataPartition (Y_attribute,... ) This func makes sure that all class levels 
#    in dept vars are represented in the test & train sets.
inTraining <- createDataPartition(wyCompOOB$brand, p=0.75, list=FALSE)
oobTrain <- wyCompOOB[inTraining,]   
oobTest <- wyCompOOB[-inTraining,]   
# verify number of obs 
nrow(oobTrain)
nrow(oobTest)  
#
library(skimr)
skimmed <- skim(oobTrain)
skimmed

#
# -------------------------------------------
# -- 2. SET COMMON PARAMS FOR ALL MODELS
# -------------------------------------------
#
# Fix the  seed for repeatability of results
set.seed(123) 
#
# SET CROSS VALIDATION:
#
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1) 

#
# -------------------------------------------
# -- 3. RUN DIFFERENT MODELS
# -------------------------------------------
#
# --- 3.1 RANDOM FOREST - OOB
# ---------------------------
#
oobRFfit <- train(brand~., data=oobTrain, method="rf", importance=T, trControl=fitControl)
#
oobRFfit
#
plot(oobRFfit)
varImp(oobRFfit)

#
# --- 3.1.1 RANDOM FOREST -  Grid
# -------------------------------
#
# 
rfGrid <- expand.grid(mtry=c(18,21,23,25, 30))  
#
manRFfit <- train(brand~.,data=oobTrain,method="rf", importance=T, trControl=fitControl, tuneGrid=rfGrid)
manRFfit
#
plot(manRFfit) 
varImp(manRFfit)

#
# --- 3.2 Gradient Boosting - OOB
# ---------------------------
#
#
oobGBfit <- train(brand~., data=oobTrain, method="gbm", verbose = FALSE, trControl=fitControl)
#
oobGBfit
#
plot(oobGBfit) 

#
# --- 3.2.1 Gradient Boosting - Grid
# --------------------------- 
#
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2), n.trees = seq(100, 1000, by = 50), shrinkage = c(0.01, 0.1), n.minobsinnode=10)
grdGBfit <- train(brand~., data=oobTrain, method="gbm", verbose = FALSE, trControl=fitControl, tuneGrid = gbmGrid)
#
grdGBfit
plot(grdGBfit) 
#varImp(obGBfit) 
#
#
# --- 3.3 C5.0 - OOB
# ------------------------------
#
library(inum)
library(C50)
#
oobC5fit <- train(brand~., data=oobTrain, method="C5.0", importance=T, trControl=fitControl) 
#
oobC5fit
#
plot(oobC5fit)
varImp(oobC5fit)
#

#
# --- 3.3.1 C5.0 - Grid
# ------------------------------
#
# I: Based on def results, model=rules is better. Try more trials to see if accuracy improves.
c5Grid <- expand.grid(.trials = seq(5, 40, by = 2), .model = "rules", .winnow = c(TRUE, FALSE))
#
grdC5fit <- train(brand~., data=oobTrain, method="C5.0", importance=T, trControl=fitControl,  tuneGrid=c5Grid)
#
grdC5fit
#
plot(grdC5fit)
varImp(grdC5fit)

#
# -------------------------------------------
# -- 4. MODEL SELECTION
# -------------------------------------------
#
ModelResample <- resamples(list(gb=grdGBfit, rf=oobRFfit))

ModelResample
# output summary metrics for tuned models 
summary(ModelResample)

parallelplot(ModelResample, auto.key=TRUE)
#

##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(grdGBfit, "grdGBfit.rds")

# load and name model
goldFit <- readRDS("grdGBfit.rds")

#
# -------------------------------------------
# -- 5. RFE: MODEL TESTING
# -------------------------------------------
# 
# --- 5.1 Test Best model
# -----------------------
#
goldTest <- predict(grdGBfit, oobTest)
#
# --- 5.2. Performace Measurment
# ------------------------------
postResample(goldTest, oobTest$brand)
#
confusionMatrix(data=goldTest, oobTest$brand)

#
#--- 5.3 Plot Results
#-----------------------
#
library(ggplot2)
#library(dplyr)

plot_confmatrix <- function(test_iv, test_dv) {
  table <- data.frame(confusionMatrix(data=test_iv, test_dv)$table)
  
  plotTable <- table %>%
    mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
    group_by(Reference) %>%
    mutate(prop = Freq/sum(Freq))
 
  ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
    scale_fill_manual(values = c(good = "green", bad = "red")) +
    theme_bw() +
    xlim(rev(levels(table$Reference)))
} # Eo plot_confmatrix

plot_confmatrix(goldTest,oobTest$brand)

# plot predicted verses actual
plot(goldTest, oobTest$brand)
#
# Receiver Operating Characteristic Curves (ROC)
# -----------------------------------------------
library(precrec)
#
precrec_obj <- evalmod(scores = rocTest, labels = reffTest)
autoplot(precrec_obj)
auc(precrec_obj)

precrec_obj2 <- evalmod(scores = rocTest, labels = reffTest, mode="basic")
autoplot(precrec_obj2)  

#
# -------------------------------------------
# -- 6. PREDCIT WITH MODE
# -------------------------------------------
#
# 6.1 Production Prediction 
# -------------------------
# predict for new dataset with no values for DV
#
goldPred <- predict(grdGBfit, wyIncompOOB)

head(goldPred)
#
# Performace Measurment
# ----------------------
postResample(goldPred, wyIncompOOB$brand)
#
confusionMatrix(data=goldPred, wyIncompOOB$brand)
#
# plot predicted verses actual
#--------------------
plot(goldPred, wyIncompOOB$brand)
#
#
Pred_Brand <- copy(goldPred)
str(Pred_Brand)

write.csv(Pred_Brand, file = "Incomp_Prediction.csv")

# ==
# =================================================
# =================================================
# ==      PREDICTIVE MODELS FOR RFE DATASET      ==
# =================================================
# =================================================
# ==
#
# 
# ---------------------------------
# -- 1. RFE: CREATE TRAIN/TEST SETS
# ---------------------------------
#
describe(wyCompRFE) # dlookr pkg
rfeTraining <- createDataPartition(wyCompRFE$brand, p=0.75, list=FALSE)
rfeTrain <- wyCompRFE[rfeTraining,]   
rfeTest <- wyCompRFE[-rfeTraining,]
#
# -------------------------------------------
# -- 2. RFE: SET COMMON PARAMS FOR ALL MODELS
# -------------------------------------------
#
# Fix the  seed for repeatability of results
set.seed(123)
#
#
# SET CROSS VALIDATION:
#
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)  # I: Var reused.
#
# -------------------------------------------
# -- 3. RFE: RUN DIFFERENT MODELS
# -------------------------------------------
#
# --- 3.1 RANDOM FOREST - OOB
# ---------------------------
#
roobRFfit <- train(brand~.,data=rfeTrain,method="rf", importance=T, trControl=fitControl)
#
roobRFfit
plot(roobRFfit) 
varImp(roobRFfit)

#
# --- 3.1.1 RANDOM FOREST -  Grid
# -------------------------------
#
rfGrid <- expand.grid(mtry=c(3,4,5,6,7,8,9,10,11,12))  
#
rgrdRFfit <- train(brand~.,data=rfeTrain,method="rf", importance=T, trControl=fitControl, tuneGrid=rfGrid)
#
rgrdRFfit
plot(rgrdRFfit)
varImp(rgrdRFfit)

#
# --- 3.2 Gradient Boosting - OOB
# ---------------------------
#
rfeGBfit <- train(brand~., data=rfeTrain, method="gbm", verbose = FALSE, trControl=fitControl)
#
rfeGBfit
plot(rfeGBfit) 

#
# --- 3.2.1 Gradient Boosting - Grid
# --------------------------- 
#
gbmrGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2), n.trees = seq(100, 1000, by = 50),
                        shrinkage = c(0.01, 0.1), n.minobsinnode=10)
#
rgrdGBfit <- train(brand~., data=rfeTrain, method="gbm", verbose = FALSE, trControl=fitControl,
                  tuneGrid = gbmrGrid)
#
rgrdGBfit
plot(rgrdGBfit) 

#
# --- 3.3 C5.0 - OOB
# ------------------------------
#
#library(inum)
#library(C50)
#
rfeC5fit <- train(brand~., data=rfeTrain, method="C5.0", importance=T, trControl=fitControl)
#
rfeC5fit
plot(rfeC5fit)
varImp(rfeC5fit)

#
# --- 3.3.1 C5.0 - Grid
# ------------------------------
#
c5Grid2 <- expand.grid(.trials = seq(2, 50, by = 2), .model = c("tree","rules"), .winnow = c(FALSE,TRUE))
#
rgrdC5fit <- train(brand~., data=rfeTrain, method="C5.0", importance=T, trControl=fitControl,
                   tuneGrid=c5Grid2)
#
rgrdC5fit
plot(rgrdC5fit)
varImp(rgrdC5fit)

#
# -------------------------------------------
# -- 4. RFE: MODEL SELECTION
# -------------------------------------------
#
# I : Choose top two winning models and compare the two models.
#
RFE_Resample <- resamples(list(gb=rgrdGBfit, c5=rgrdC5fit)) #AI: update model(s)
#
#   Output summary metrics for tuned models 
summary(RFE_Resample)
#
#   Plot the RMSE values
parallelplot(RFE_Resample)
#
# I: Save top performing model after validation
saveRDS(rgrdGBfit, "RFE_gold_fit.rds")  #AI: update model(s)
#
# -------------------------------------------
# -- 5. RFE: MODEL TESTING
# -------------------------------------------
# 
# --- 5.1 Test Best model
# -----------------------
str(rfeTest)
#
rfe_goldTest   <- predict(rgrdGBfit, rfeTest)  
rfe_silverTest <- predict(rgrdC5fit, rfeTest)
#
# --- 5.2. Performace Measurment
# ------------------------------
postResample(rfe_goldTest, rfeTest$brand)
#
postResample(rfe_silverTest, rfeTest$brand)
#
confusionMatrix(data=rfe_goldTest, rfeTest$brand)
plot_confmatrix(rfe_goldTest, rfeTest$brand)

#
#--- 5.3 Plot Results
#-----------------------
#
# Plot predicted verses actual
plot(rfe_goldTest, rfeTest$brand)
#
# I: Receiver Operating Characteristic Curves (ROC)
library(precrec)
#
#    Numeric predictor is required by roc().
rocTest <- as.numeric(rfe_goldTest)
rocTest[rocTest==1] <- 0
rocTest[rocTest==2] <- 1
rocTest
#
#   Change reff set to numeric as well (MR).
reffTest <- as.numeric(rfeTest$brand)
reffTest[reffTest==1] <- 0
reffTest[reffTest==2] <- 1
reffTest
#
#   FYI: as.factor(rfeTest$brand) Use as levels by roc()
precrec_obj <- evalmod(scores = rocTest, labels = reffTest)
autoplot(precrec_obj)
auc(precrec_obj)

#
# -------------------------------------------
# -- 6. RFE: PREDCIT WITH MODE
# -------------------------------------------
#
# 6.1 Production Prediction 
# -------------------------
#
rfe_goldPred <- predict(rgrdGBfit, wyIncompOOB) #AI: update model(s)
#
head(rfe_goldPred)
#
# Performace Measurment (per task reqs)
# ----------------------
postResample(rfe_goldPred, wyIncompOOB$brand)
#
confusionMatrix(data=rfe_goldPred, wyIncompOOB$brand)
#
plot(rfe_goldPred, wyIncompOOB$brand)
