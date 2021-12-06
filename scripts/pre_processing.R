# Pacotes -----------------------------------------------------------------
library(readr)
library(tidyverse)

# Lendo a base ------------------------------------------------------------
heart <- read_csv("data/heart.csv") |>
  janitor::clean_names() #ajustando o nome das variáveis

# Variável resposta - Heart Disease ---------------------------------------
heart |>
  ggplot(mapping = aes(x = heart_disease, fill = heart$sex)) +
  geom_bar()

heart |> count(heart_disease)

# Features ----------------------------------------------------------------
glimpse(heart) # base bruta

heart_adjust <- heart |>
  mutate(
    # ajustando o sexo
    sex = case_when(sex == "M" ~ 0,
                    sex == "F" ~ 1),
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
  mutate(heart_disease = as.factor(heart_disease)
         )

glimpse(heart_adjust)

# Exploratória ------------------------------------------------------------
skimr::skim(heart_adjust)

visdat::vis_miss(heart_adjust) # missing data

heart_adjust %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs") %>%
  corrplot::corrplot()

# Nova base de dados ------------------------------------------------------
saveRDS(object = heart_adjust,file = "base.rds")
