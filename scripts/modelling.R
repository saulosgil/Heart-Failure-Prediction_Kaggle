# Pacotes ------------------------------------------------------------------

library(tidymodels)
library(tidyverse)
library(pROC)
library(vip)
library(randomForest)


# Bases de dados ----------------------------------------------------------
base <- readr::read_rds("base.rds")

glimpse(base)

base |> count(heart_disease)

# Base de treino e teste --------------------------------------------------

set.seed(1)
base_initial_split <- initial_split(base, strata = "heart_disease", prop = 0.80)

base_train <- training(base_initial_split)
base_test  <- testing(base_initial_split)

# Reamostragem ------------------------------------------------------------

base_resamples <- vfold_cv(base_train, v = 5, strata = "heart_disease")

# Exploratória ------------------------------------------------------------

skimr::skim(base_train)

visdat::vis_miss(base_train)

base_train |>
   select(where(is.numeric)) |>
   cor(use = "pairwise.complete.obs") |>
   corrplot::corrplot()


# Regressão Logística -----------------------------------------------------

## Data prep
base_lr_recipe <- recipe(heart_disease ~ ., data = base_train) |>
  step_normalize(all_numeric_predictors()) |>
  # step_novel(all_nominal_predictors()) |>
  # step_impute_linear(Income, Assets, Debt, impute_with = imp_vars(Expenses)) %>%
  # step_impute_mode(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_poly(all_numeric_predictors(), degree = 1) |>
  step_corr(all_numeric_predictors())
  # step_dummy(all_nominal_predictors())

bake(prep(base_lr_recipe), new_data = NULL)

visdat::vis_miss(bake(prep(base_lr_recipe), new_data = NULL))

## Modelo

base_lr_model <- logistic_reg(
  penalty = tune(),
  mixture = 1) |>
  set_mode("classification") |>
  set_engine("glmnet")

## Workflow

base_lr_wf <- workflow() |>
  add_model(base_lr_model) |>
  add_recipe(base_lr_recipe)

## Tune

grid_lr <- grid_regular(
  penalty(range = c(-4, -1)),
  levels = 20
)

# doParallel::registerDoParallel(4)

base_lr_tune_grid <- tune_grid(
  base_lr_wf,
  resamples = base_resamples,
  grid = grid_lr,
  metrics = metric_set(roc_auc)
)

autoplot(base_lr_tune_grid)
collect_metrics(base_lr_tune_grid)

# Árvore de decisão -------------------------------------------------------

## Data prep

base_dt_recipe <- recipe(heart_disease ~ ., data = base) |>
 # step_novel(all_nominal_predictors()) %>%
  step_zv(all_predictors()) |>
  step_corr(all_numeric_predictors())

## Modelo

base_dt_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) |>
  set_mode("classification") |>
  set_engine("rpart")

## Workflow

base_dt_wf <- workflow() |>
  add_model(base_dt_model) |>
  add_recipe(base_dt_recipe)

## Tune

grid_dt <- grid_random(
  cost_complexity(c(-9, -2)),
  tree_depth(range = c(5, 15)),
  min_n(range = c(20, 40)),
  size = 20
)

# doParallel::registerDoParallel(4)

base_dt_tune_grid <- tune_grid(
  base_dt_wf,
  resamples = base_resamples,
  grid = grid_dt,
  metrics = metric_set(roc_auc)
)

# doParallel::stopImplicitCluster()

autoplot(base_dt_tune_grid)
collect_metrics(base_dt_tune_grid)

# Random forest ---------------------------------------------------------

## Data prep

base_rf_recipe <- recipe(heart_disease ~ ., data = base) |>
  step_zv(all_predictors()) |>
  step_corr(all_numeric_predictors())

## Modelo
base_rf_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_mode("classification") |>
  set_engine("ranger")

## Workflow

base_rf_wf <- workflow() |>
  add_model(base_rf_model) |>
  add_recipe(base_rf_recipe)

## Tune

grid_rf <- grid_random(
  min_n(range = c(5, 25)),
  mtry(range = c(10, 18)),
  trees(range = c(20, 400))
)

# doParallel::registerDoParallel(4)

base_rf_tune_grid <- tune_grid(
  base_rf_wf,
  resamples = base_resamples,
  grid = grid_rf,
  metrics = metric_set(roc_auc)
)

# doParallel::stopImplicitCluster()

autoplot(base_rf_tune_grid)
collect_metrics(base_rf_tune_grid)

# Desempenho dos modelos finais ----------------------------------------------

base_lr_best_params <- select_best(base_lr_tune_grid, "roc_auc")
base_lr_wf <- base_lr_wf |> finalize_workflow(base_lr_best_params)
base_lr_last_fit <- last_fit(base_lr_wf, base_initial_split)

base_dt_best_params <- select_best(base_dt_tune_grid, "roc_auc")
base_dt_wf <- base_dt_wf |>  finalize_workflow(base_dt_best_params)
base_dt_last_fit <- last_fit(base_dt_wf, base_initial_split)

base_rf_best_params <- select_best(base_rf_tune_grid, "roc_auc")
base_rf_wf <- base_rf_wf |>  finalize_workflow(base_rf_best_params)
base_rf_last_fit <- last_fit(base_rf_wf, base_initial_split)

base_test_preds <- bind_rows(
  collect_predictions(base_lr_last_fit) |> mutate(modelo = "LinearRegression"),
  collect_predictions(base_dt_last_fit) |> mutate(modelo = "DecisionTree"),
  collect_predictions(base_rf_last_fit) |> mutate(modelo = "RandomForest")
)

## roc
base_test_preds |>
  group_by(modelo) |>
  roc_curve(heart_disease, .pred_0) |>
  autoplot()

## lift
base_test_preds %>%
  group_by(modelo) %>%
  lift_curve(heart_disease, .pred_0) %>%
  autoplot()

# Variáveis importantes
base_lr_last_fit_model <- base_lr_last_fit$.workflow[[1]]$fit$fit
vip(base_lr_last_fit_model)

base_dt_last_fit_model <- base_dt_last_fit$.workflow[[1]]$fit$fit
vip(base_dt_last_fit_model)

# base_rf_last_fit_model <- base_rf_last_fit$.workflow[[1]]$fit$fit
# vip(base_rf_last_fit_model)

# Guardar tudo ------------------------------------------------------------

write_rds(base_lr_last_fit, "base_lr_last_fit.rds")
write_rds(base_lr_model, "base_lr_model.rds")


# Modelo final ------------------------------------------------------------

base_final_lr_model <- base_lr_wf |> fit(base)

test1 <- base[4,]

predict(base_final_lr_model, new_data = base)





