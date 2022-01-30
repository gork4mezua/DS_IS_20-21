##----------------------------------------------------------------
## Subject: Intelligent Systems (2021-2022)                      -
## Topic: NPL Practical work                                     -
## Author: Gorka Amezua Lasuen                                   -
## Last updated: 28/01/2022                                      -
##----------------------------------------------------------------
# [0] Loads necessary libraries
library(bannerCommenter)
library(utf8)
library(spacyr)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(stopwords)
library(dplyr)
library(tidytext)
library(quanteda)
library(quanteda.textmodels)
library(caret)
library(superml)

# [1] Establishes this files directory as working directory
curdir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curdir)

# [2] Reads the csv files (train and test)
dftrain <- read.csv("data/SMS_train.csv", header = TRUE)
dftest <- read.csv("data/SMS_test.csv", header = TRUE)

# [3] Sets data types
# [3.1] Training file
dftrain$Label <- as.factor(dftrain$Label) 
# [3.1] Test file
dftest$Label <- as.factor(dftest$Label) 

# [4] Preprocessing
# [4.1] Remove unused columns
dftrain$S..No. <- NULL
dftest$S..No. <- NULL
# [4.2] Check all Message bodies maade by UTF8 characters
# [4.2.1] Train set
check <- rep(TRUE, nrow(dftrain))
res <- all.equal(utf8_valid(dftrain$Message_body), check)
text <- paste("Training set (UTF8 characters): ", res)
banner(text, centre = TRUE, bandChar = "-")
# [4.2.2] Test set
check <- rep(TRUE, nrow(dftest))
res <- all.equal(utf8_valid(dftest$Message_body), check)
text <- paste("Test set (UTF8 characters): ", res)
banner(text, centre = TRUE, bandChar = "-")
# [4.3] Check all character normalization
# [4.3.1] Train set
nrm <- utf8_normalize(dftrain$Message_body)
res <- sum(nrm != dftrain$Message_body)
text <- paste("Training set (UTF8 normalized): ", res)
banner(text, centre = TRUE, bandChar = "-")
# [4.3.2] Test set
nrm <- utf8_normalize(dftest$Message_body)
res <- sum(nrm != dftest$Message_body)
text <- paste("Test set (UTF8 normalized): ", res)
banner(text, centre = TRUE, bandChar = "-")
# [4.4] Basic cleaning (replace line break and double spaces by single space)
# [4.4.1] Train set
dftrain$Message_body <- gsub("[\n]{1,}", " ", dftrain$Message_body) 
dftrain$Message_body <- gsub("[ ]{2,}", " ", dftrain$Message_body)
# [4.4.2] Test set
dftest$Message_body <- gsub("[\n]{1,}", " ", dftest$Message_body)
dftest$Message_body <- gsub("[ ]{2,}", " ", dftest$Message_body)
# [4.5] Advanced cleaning (remove special characters, digits, urls, single characters)
# [4.5.1] Train set
dftrain$Message_body <- gsub("[#,@,&]", "", dftrain$Message_body)
dftrain$Message_body <- gsub("[0-9]+", "", dftrain$Message_body) 
dftrain$Message_body <- gsub("w{3}", "", dftrain$Message_body) 
dftrain$Message_body <- gsub(" ?(f|ht)tp(s?)://(.*)[.][a-z]+", "", dftrain$Message_body)
dftrain$Message_body <- gsub("\\W*\\b\\w\\b\\W*", "", dftrain$Message_body)  
# [4.5.2] Test set
dftest$Message_body <- gsub("[#,@,&]", "", dftest$Message_body)
dftest$Message_body <- gsub("[0-9]+", "", dftest$Message_body) 
dftest$Message_body <- gsub("w{3}", "", dftest$Message_body) 
dftest$Message_body <- gsub(" ?(f|ht)tp(s?)://(.*)[.][a-z]+", "", dftest$Message_body)
dftest$Message_body <- gsub("\\W*\\b\\w\\b\\W*", "", dftest$Message_body)
# [4.6] Convert to lowercase
# [4.6.1] Train set
dftrain$Message_body <- tolower(dftrain$Message_body)
# [4.6.2] Test set
dftest$Message_body <- tolower(dftest$Message_body)  
# [4.7] Encode labels
label <- LabelEncoder$new()
dftrain$Label <- label$fit_transform(dftrain$Label)
dftest$Label <- label$fit_transform(dftest$Label)

# [5] Tokenize (spaCy)
spacy_initialize(model = "en_core_web_sm")
# [5.1] Train set
traintok <- spacy_tokenize(dftrain$Message_body,
    what="sentence",
    remove_punct = TRUE,
    remove_url = TRUE,
    remove_numbers = TRUE,
    remove_separators = TRUE,
    remove_symbols = TRUE,
    padding = FALSE,
    multithread = TRUE,
    output = "list"
)
v_traintok <- unlist(traintok)
#v_traintok <- v_traintok[-which(v_traintok=="")] # Remove empty sentences
numsen_train <- length(v_traintok)
text <- paste("Num. sentences (train): ", numsen_train)
banner(text, centre = TRUE, bandChar = "-")
# Histogram for sentences size
hist(nchar(v_traintok),
    main = "Histogram of sentence size",
    xlab = "Sentece size (number of characters)",
    ylab = "Ocurrences"
)
# [5.2] Test set
testtok <- spacy_tokenize(dftest$Message_body,
    what="sentence",
    remove_punct = TRUE,
    remove_url = TRUE,
    remove_numbers = TRUE,
    remove_separators = TRUE,
    remove_symbols = TRUE,
    padding = FALSE,
    multithread = TRUE,
    output = "list"
)
v_testtok <- unlist(testtok)
#v_testtok <- v_testtok[-which(v_testtok=="")] # Remove empty sentences
numsen_test <- length(v_testtok)
text <- paste("Num. sentences (test): ", numsen_test)
banner(text, centre = TRUE, bandChar = "-")
# Histogram for sentences size
hist(nchar(v_testtok),
    main = "Histogram of sentence size",
    xlab = "Sentece size (number of characters)",
    ylab = "Ocurrences"
)


# [6] Create and show spam wordcloud
# Create stopwords variable
stopwords = stopwords("en")
names(stopwords) = "word"
# [6.1] Train set
trainsub <- subset(dftrain, Label=='1')
trainword <- spacy_tokenize(trainsub$Message_body,
    what="word",
    remove_punct = TRUE,
    remove_url = TRUE,
    remove_numbers = TRUE,
    remove_separators = TRUE,
    remove_symbols = TRUE,
    padding = FALSE,
    multithread = TRUE,
    output = "list"
)
v_trainword <- unlist(trainword)
trainrank <- sort(table(v_trainword),decreasing=TRUE) 
dftrainrank <- data.frame(word = names(trainrank),freq=as.numeric(trainrank))
# Remove stopwords
dftrainrank <- anti_join(dftrainrank, stop_words, by="word")
# Create map
set.seed(1234)
wordcloud(words = dftrainrank$word, freq = dftrainrank$freq, min.freq = 1,           
    max.words=200, random.order=FALSE, rot.per=0.35, excludeWords=stopwords::stopwords("en"),
    colors=brewer.pal(8, "Dark2"))
# [6.2] Test set
testsub <- subset(dftest, Label=='1')
testword <- spacy_tokenize(testsub$Message_body,
    what="word",
    remove_punct = TRUE,
    remove_url = TRUE,
    remove_numbers = TRUE,
    remove_separators = TRUE,
    remove_symbols = TRUE,
    padding = FALSE,
    multithread = TRUE,
    output = "list"
)
v_testword <- unlist(testword)
testrank <- sort(table(v_testword),decreasing=TRUE) 
dftestrank <- data.frame(word = names(testrank),freq=as.numeric(testrank))
# Remove stopwords
dftestrank <- anti_join(dftestrank, stop_words, by="word")
# Create map
set.seed(1234)
wordcloud(words = dftestrank$word, freq = dftestrank$freq, min.freq = 1,           
    max.words=200, random.order=FALSE, rot.per=0.35,
    colors=brewer.pal(8, "Dark2"))

# [7] Analysis and Modeling
tic <- Sys.time()
# [7.1] Generate the corpus
# [7.1.1] Train set
corptrain <- corpus(dftrain, text_field = "Message_body")
# Save result on a file
saveRDS(corp, file="spacy-trainparse.rds")
# [7.1.2] Test set
corptest <- corpus(dftest, text_field = "Message_body")
# Save result on a file
saveRDS(corptest, file="spacy-testparse.rds")
# Calculate duration of process
Sys.time()-tic
# [7.2] Generate the model
# Create dfm objects
dfmtrain <- dfm(corptrain, remove=stopwords("en"))
dfmtest <- dfm(corptest=stopwords("en"))
# Create model
model <- textmodel_nb(dfmtrain, docvars(dfmtrain, "Label"), prior = "docfreq")
dfmat_matched <- dfm_match(dfmtest, features = featnames(dfmtrain))
# Predicting with testing df
actual_class <- docvars(dfmat_matched, "Label")
predicted_class <- predict(model, newdata = dfmat_matched)
tab_class <- table(actual_class, predicted_class)
# Confusion matrix
confusion <- confusionMatrix(tab_class, mode = "everything")
# Plot contingency table
# Save confusion matrix as data frame
confusion.data <- as.data.frame(confusion[["table"]])
# Reverse the order
level_order_y <-
  factor(confusion.data$actual_class,
         level = c('1', '0'))
ggplot(confusion.data,
       aes(x = predicted_class, y = level_order_y, fill = Freq)) +
  xlab("Predicted class") +
  ylab("Actual class") +
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette = "Blues", direction = 1) +
  scale_x_discrete(labels = c("Non-Spam", "Spam")) +
  scale_y_discrete(labels = c("Spam", "Non-Spam"))

# [7.3] Finish spacy session
spacy_finalize()
sessionInfo()