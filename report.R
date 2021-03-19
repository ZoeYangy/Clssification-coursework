
heart_data <- readr::read_csv("D:\\Durham\\Machine_Learning\\Classification\\Coursework\\summative\\heart_failure.csv")
skimr::skim(heart_data)
View(heart_data)
# correlation analysis
library(corrplot)
library(dplyr)
corr <- cor(heart_data)
corr <- corrplot(corr,method = 'number')
heart_data <- heart_data%>%
  select(-diabetes, -sex)# delete corr=0 features
dim(heart_data)
# Plots
DataExplorer::plot_bar(heart_data, ncol = 3)
DataExplorer::plot_histogram(heart_data, ncol = 3)
DataExplorer::plot_boxplot(heart_data, by = "fatal_mi", ncol = 3)
#Task and resampling
library("data.table")
library("mlr3verse")
library("mlr3viz")
set.seed(212)
# data split
library("rsample")
set.seed(212) # by setting the seed we know everyone will see the same results
# First get the training
heart_split <- initial_split(heart_data)
heart_train <- training(heart_split)
# Then further split the training into validate and test
heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)
# 
#define task
heart_data$fatal_mi <- as.factor(heart_data$fatal_mi)
heart_task <- TaskClassif$new(id = "fatal.detect",
                               backend = heart_data,
                               target = 'fatal_mi',
                               positive = '0')
# define resampling
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)# »á·Ö³ötrainºÍtest
cv5$instantiate(heart_task)

#train = list(c(1:10, 51:60, 101:110)),
#test = list(c(11:20, 61:70, 111:120))
str(cv5$train_set(1))
str(cv5$test_set(1))
#define learners
#log_reg
lrn_log <- lrn("classif.log_reg", predict_type = "prob")
#tree
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.03)
lrn_log$param_set
lrn_cart$param_set

#fit, not use benchmark
res.log <- resample(heart_task, lrn_log, cv5, store_models = TRUE)
res.cart <- resample(heart_task, lrn_cart_cp, cv5, store_models = TRUE)
# log.prediction = res.log$prediction()

#fit. use benchmark
res <- benchmark(data.table(
  task = list(heart_task),
  learner = list(lrn_log,
                 #lrn_cart,
                 lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

#res$aggregate()
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
#ROC
autoplot(res.log, type = 'roc')
autoplot(res.cart, type = "roc")

#TUNE
library("paradox")
search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)
library("mlr3tuning")
evals20 = trm("evals", n_evals = 20)
instance = TuningInstanceSingleCrit$new(
  task = heart_task,
  learner = lrn_cart_cp,
  resampling = cv5,
  measure = msr("classif.ce"),
  search_space = search_space,
  terminator = evals20
)

tuner = tnr("grid_search", resolution = 5)
tuner$optimize(instance)
instance$result_learner_param_vals
instance$result_y

cp = instance$result_learner_param_vals$cp
minsplit = instance$result_learner_param_vals$minsplit
lrn_cart_tune <- lrn("classif.rpart", predict_type = "prob", cp = cp, minsplit = minsplit)
res_tune <- benchmark(data.table(
  task = list(heart_task),
  learner = list(lrn_cart_cp,
                 lrn_cart_tune),
  resampling = list(cv5)
), store_models = TRUE)
res_tune$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
# tree's improvements
res.tree <- benchmark(data.table(
  task = list(heart_task),
  learner = list(lrn_cart,
                 lrn_cart_cp,
                 lrn_cart_tune),
  resampling = list(cv5)
), store_models = TRUE)
res.tree$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.auc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))


# Plot tree
trees <- res_tune$resample_result(2)
#the tree from first CV iteration
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

autoplot(res.log, type = 'roc')
autoplot(res.cart, type = "roc")






# deep learning
library("rsample")
set.seed(212) # by setting the seed we know everyone will see the same results
# First get the training
heart_split <- initial_split(heart_data)
heart_train <- training(heart_split)
# Then further split the training into validate and test
heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)

# all numeric, no missing data
library("recipes")
cake <- recipe(fatal_mi~ ., data = heart_data) %>%
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = heart_train) # learn all the parameters of preprocessing on the training data

heart_train_final <- bake(cake, new_data = heart_train) # apply preprocessing to training data
heart_validate_final <- bake(cake, new_data = heart_validate) # apply preprocessing to validation data
heart_test_final <- bake(cake, new_data = heart_test) # apply preprocessing to testing data

library("keras")
heart_train_x <- heart_train_final %>%
  select(-starts_with("fatal_mi")) %>%
  as.matrix()
heart_train_y <- heart_train_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_validate_x <- heart_validate_final %>%
  select(-starts_with("fatal_mi")) %>%
  as.matrix()
heart_validate_y <- heart_validate_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_test_x <- heart_test_final %>%
  select(-starts_with("fatal_mi")) %>%
  as.matrix()
heart_test_y <- heart_test_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()


deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_x))) %>%
  #layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#deeper
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu",
              input_shape = c(ncol(heart_train_x))) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  heart_train_x, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_x, heart_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict_proba(heart_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict_classes(heart_test_x)

table(pred_test_res, heart_test_y)
yardstick::accuracy_vec(as.factor(heart_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(heart_test_y, levels = c("1","0")),
                       c(pred_test_prob))

