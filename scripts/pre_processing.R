# Pacotes -----------------------------------------------------------------
library(readr)
library(tidyverse)
library(tidymodels)
library(GGally)

# Lendo a base ------------------------------------------------------------
heart <- read_csv("data/heart.csv") |>
  janitor::clean_names() #ajustando o nome das variáveis

# Variável resposta - Heart Disease ---------------------------------------
heart |>
  ggplot(mapping = aes(x = heart_disease, fill = sex)) +
  geom_bar()

heart |> count(heart_disease)

# Features ----------------------------------------------------------------
glimpse(heart) # base bruta

heart_adjust <- heart |>
  mutate(
    # ajustando o sexo
    sex = case_when(sex == "M" ~ "0",
                    sex == "F" ~ "1"),
    # criando 4 dummies para chest_pain_type que apresenta 4 categorias
    chest_pain_type_ATA = case_when(chest_pain_type == "ATA" ~ 1,
                                    chest_pain_type != "ATA" ~ 0),
    chest_pain_type_NAP = case_when(chest_pain_type == "NAP" ~ 1,
                                    chest_pain_type != "NAP" ~ 0),
    chest_pain_type_ASY = case_when(chest_pain_type == "ASY" ~ 1,
                                    chest_pain_type != "ASY" ~ 0),
    chest_pain_type_TA = case_when(chest_pain_type == "TA" ~ 1,
                                    chest_pain_type != "TA" ~ 0),
    # criando 3 dummies para resting_ecg que apresenta 3 categorias
    resting_ecg_normal = case_when(resting_ecg == "Normal" ~ 1,
                                   resting_ecg != "Normal" ~ 0),
    resting_ecg_st = case_when(resting_ecg == "ST" ~ 1,
                               resting_ecg != "ST" ~ 0),
    resting_ecg_lvh = case_when(resting_ecg == "LVH" ~ 1,
                                 resting_ecg != "LVH" ~ 0),
    # criando 2 dummies para exercise_angina que apresenta 2 categorias
    exercise_angina = case_when(exercise_angina == "Y" ~ 1,
                                exercise_angina == "N" ~ 0),
    # criando 3 dummies para st_slope que apresenta 2 categorias
    st_slope_up = case_when(st_slope == "Up" ~ 1,
                            st_slope != "Up" ~ 0),
    st_slope_flat = case_when(st_slope == "Flat" ~ 1,
                            st_slope != "Flat" ~ 0),
    st_slope_down = case_when(st_slope == "Down" ~ 1,
                            st_slope != "Down" ~ 0)
    ) |>
  # selecionando as features
  select(age,
         sex,
         resting_bp,
         cholesterol,
         fasting_bs,
         max_hr,
         oldpeak,
         chest_pain_type_ATA,
         chest_pain_type_NAP,
         chest_pain_type_ASY,
         chest_pain_type_TA,
         resting_ecg_normal,
         resting_ecg_st,
         resting_ecg_lvh,
         exercise_angina,
         st_slope_up,
         st_slope_flat,
         st_slope_down,
         heart_disease
         ) |>
  mutate(sex = as.character(sex),
         fasting_bs = as.character(fasting_bs),
         chest_pain_type_ATA = as.character(chest_pain_type_ATA),
         chest_pain_type_NAP = as.character(chest_pain_type_NAP),
         chest_pain_type_ASY = as.character(chest_pain_type_ASY),
         chest_pain_type_TA = as.character(chest_pain_type_TA),
         resting_ecg_normal = as.character(resting_ecg_normal),
         resting_ecg_st = as.character(resting_ecg_st),
         resting_ecg_lvh = as.character(resting_ecg_lvh),
         exercise_angina = as.character(exercise_angina),
         st_slope_up = as.character(st_slope_up),
         st_slope_flat = as.character(st_slope_flat),
         st_slope_down = as.character(st_slope_down),
         heart_disease = as.character(heart_disease)
         )

glimpse(heart_adjust)

# Exploratória ------------------------------------------------------------
skimr::skim(heart_adjust)

visdat::vis_miss(heart_adjust) # missing data

# Correlações - heart disease vs numerics

heart_adjust |>
  select(age,
         resting_bp,
         cholesterol,
         max_hr,
         oldpeak) |>
  cor(use = "pairwise.complete.obs") |>
  corrplot::corrplot()

heart_adjust |>
  select(c(where(is.numeric), heart_disease)) |>
  ggpairs(aes(colour = heart_disease))

# Descritivas - vars vs heart disease

contagens_numer <- heart_adjust |>
  select(c(where(is.numeric)),
         heart_disease) |>
  pivot_longer(-heart_disease, names_to = "variavel", values_to = "valor") %>%
  count(heart_disease, variavel, valor)

# Descritivas - heart_disease vs numerics

contagens_numer |>
  ggplot(aes(y = heart_disease, x = valor, fill = heart_disease)) +
  geom_boxplot() +
  facet_wrap(~variavel, scales = "free_x") +
  # scale_x_log10() +
  ggtitle("heart_disease vs. Variáveis Numéricas")

# Descritivas - heart_disease vs categóricas

contagens_categ <- heart_adjust |>
  select(c(!where(is.numeric)),
         heart_disease) |>
  pivot_longer(-heart_disease, names_to = "variavel", values_to = "valor") |>
  count(heart_disease, variavel, valor)

contagens_categ |>
  ggplot(aes(y = valor, x = n, fill = heart_disease)) +
  geom_col(position = "fill") +
  geom_label(aes(label = n), position = position_fill(vjust = 0.5)) +
  facet_wrap(~variavel, scales = "free_y", ncol = 3) +
  ggtitle("heart_disease vs. Variáveis Categóricas")

# Nova base de dados ------------------------------------------------------
saveRDS(object = heart_adjust,file = "data/base.rds")

