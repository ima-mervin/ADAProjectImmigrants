# ================================================================= #
#  1. SETUP: LOAD LIBRARIES
# ================================================================= #
# install.packages(c("tidyverse", "ipred", "caret", "e1071", "stringr", "lubridate", "randomForest"))

library(tidyverse)
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(ipred)
library(caret)
library(stringr)
library(lubridate)
library(rpart.plot)
library(rpart)
library(randomForest) # Added for the Random Forest model


# ================================================================= #
#  2. DATA CLEANING (Revised and Streamlined)
# ================================================================= #

# Load the dataset, replacing blanks and "--" with NA
# Note: Update the file path to match the location on your computer.
games <- read_csv("Downloads/game_details.csv", na = c("", "--"))

# --- DEBUGGING STEP ---
# The following line will print the exact column names from your CSV file.
cat("\n--- Column Names Read from CSV ---\n")
print(colnames(games))
cat("---------------------------------\n\n")


games <- games %>%
  mutate(
    highest_price = str_replace_all(highest_price, "€", "") %>% as.numeric(),
    release_date = as.Date(release_date, format = "%b %d, %Y"),
    only_year = lubridate::year(release_date)
  ) %>%
  dplyr::filter(only_year >= 2007)


cat("--- Missing Values Per Column After Cleaning ---\n")
print(colSums(is.na(games)))
cat("\nNumber of rows after initial cleaning:", nrow(games), "\n")


# ================================================================= #
#  3. EXPLORATORY ANALYSIS & VISUALIZATION
# ================================================================= #

# --- Question 1: What are the most popular game genres? ---

# Split genres like "Action / Shooter" into separate rows
split_genres <- games %>%
  filter(!is.na(genre)) %>%
  separate_rows(genre, sep = "\\s*/\\s*")

# Get a count of unique individual genres
total_unique_genres <- split_genres %>%
  summarise(unique_genre_count = n_distinct(genre))
cat("\nThe total unique number of individual genres is:", total_unique_genres$unique_genre_count, "\n")

# Plot the Top 10 Most Common Game Genres
top_10_genres_plot <- split_genres %>%
  count(genre, sort = TRUE) %>%
  slice_head(n = 10) %>%
  ggplot(aes(x = n, y = reorder(genre, n), fill = genre)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Top 10 Most Common Game Genres",
    subtitle = "Counts are based on splitting combined genre strings",
    x = "Number of Games",
    y = "Genre"
  ) +
  theme_minimal() +
  guides(fill = "none")

print(top_10_genres_plot)


# --- Question 2: Is there a relationship between a game's genre and its price? ---

cat("\n--- 2. Relationship Between Genre and Price (Linear Regression) ---\n")
# Note: This model is for exploration. ANOVA is used for significance testing later.
price_genre_model <- lm(highest_price ~ genre + onlyonly_year, data = games)
print(summary(price_genre_model))


# --- Question 3: How has the distribution of genres on PlayStation changed over time? ---
cat("\n--- 3. Distribution of Genres Over Time ---\n")
genre_trends <- games %>%
  filter(!is.na(onlyonly_year)) %>%
  separate_rows(genre, sep = "\\s*/\\s*") %>%
  mutate(
    genre = str_to_title(str_trim(genre)),
    genre = case_when( #had to ask AI because it wasn't showing one specific genre
      str_detect(genre, "Role.?Playing|Rpg") ~ "Role Playing Game",
      TRUE ~ genre
    )
  ) %>%
  count(only_year, genre) %>%
  group_by(genre) %>%
  filter(sum(n) > 20) %>%
  ungroup()

top_10_genres <- c("Action", "Adventure", "Role Playing Game", "Shooter", "Puzzle",
                   "Arcade", "Strategy", "Unique", "Horror", "Racing")

genre_trends_top10 <- genre_trends %>%
  filter(genre %in% top_10_genres)

# Create and print the plots
plot1 <- ggplot(filter(genre_trends_top10, genre %in% top_10_genres[1:5]),
                aes(x = only_year, y = n, group = genre, color = genre)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ genre, scales = "free_y", ncol = 2) +
  labs(title = "PlayStation Game Releases Over Time (Group 1)", y = "Number of Games Released", x = "Year") +
  theme_light() + theme(legend.position = "none")

plot2 <- ggplot(filter(genre_trends_top10, genre %in% top_10_genres[6:10]),
                aes(x = onlyonly_year, y = n, group = genre, color = genre)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ genre, scales = "free_y", ncol = 2) +
  labs(title = "PlayStation Game Releases Over Time (Group 2)", y = "Number of Games Released", x = "Year") +
  theme_light() + theme(legend.position = "none")

print(plot1)
print(plot2)


# ================================================================= #
#  4. STATISTICAL INFERENCE (ANOVA TESTS)
# ================================================================= #

# --- Question 1: Are there significant differences in user ratings across genres? ---
cat("\n--- ANOVA Test: Ratings vs. Genre ---\n")
ratings_df <- games %>%
  filter(!is.na(playstation_score), !is.na(genre), playstation_score > 0) %>%
  separate_rows(genre, sep = "\\s*/\\s*")

anova_ratings <- aov(playstation_score ~ genre, data = ratings_df)
p_value_ratings <- summary(anova_ratings)[[1]][["Pr(>F)"]][1]
cat("P-value:", p_value_ratings, "\n")
if (p_value_ratings < 0.05) { cat("Result: Reject H0 — Significant differences in ratings among genres.\n") } else { cat("Result: Fail to reject H0.\n") }


# --- Question 2: Does price vary significantly between genres? ---
cat("\n--- ANOVA Test: Price vs. Genre ---\n")
price_df <- games %>%
  filter(!is.na(highest_price), !is.na(genre), highest_price > 0) %>%
  separate_rows(genre, sep = "\\s*/\\s*")

anova_price <- aov(log1p(highest_price) ~ genre, data = price_df)
p_value_price <- summary(anova_price)[[1]][["Pr(>F)"]][1]
cat("P-value:", p_value_price, "\n")
if (p_value_price < 0.05) { cat("Result: Reject H0 — Game prices differ significantly by genre.\n") } else { cat("Result: Fail to reject H0.\n") }


# ================================================================= #
#  SPLITTING SETS + NEW CATEGORY
# ================================================================= #

# add the 'price_category' target variable to the main dataframe.
price_breaks <- quantile(games$highest_price, probs = c(0, 0.33, 0.66, 1.0), na.rm = TRUE)
games_ml <- games %>%
  mutate(price_category = cut(highest_price,
                              breaks = price_breaks,
                              labels = c("Low", "Medium", "High"),
                              include.lowest = TRUE))

# testing and training set
games_train <- games_ml %>% dplyr::filter(onlyonly_year <= 2021)
games_test  <- games_ml %>% dplyr::filter(onlyonly_year >= 2022)

#  select  predictor variables (only_year, playstation_score, genre) and our target (price_category).
# split multi-genre games and remove any rows with missing data to ensure the models run correctly.
train_data_class <- games_train %>%
  select(only_year, playstation_score, genre, price_category) %>%
  separate_rows(genre, sep = "\\s*/\\s*") %>%
  na.omit()

test_data_class <- games_test %>%
  select(only_year, playstation_score, genre, price_category) %>%
  separate_rows(genre, sep = "\\s*/\\s*") %>%
  na.omit()

#master list of all possible genres from the combined data.
all_genres <- unique(c(train_data_class$genre, test_data_class$genre))

# Now, convert 'genre' to a factor in both sets, ensuring they use the same levels.
train_data_class$genre <- factor(train_data_class$genre, levels = all_genres)
test_data_class$genre <- factor(test_data_class$genre, levels = all_genres)


# ================================================================= #
#  BAGGING
# ================================================================= #
cat("\n--- Training Bagging Classification Model ---\n")
set.seed(123) # for reproducibility
bagging_model <- bagging(price_category ~ ., # `.` means 'use all other columns as predictors'
                         data = train_data_class,
                         nbagg = 50)
print(bagging_model)

cat("\n--- Evaluating Bagging Model Performance ---\n")
predictions_bagging <- predict(bagging_model, newdata = test_data_class)
results_bagging <- confusionMatrix(predictions_bagging, test_data_class$price_category)
print(results_bagging)

# ================================================================= #
# RANDOM FOREST
# ================================================================= #

cat("\n--- Training Random Forest Model ---\n")
set.seed(123)
rf_model <- randomForest(price_category ~ .,
                         data = train_data_class,
                         ntree = 100) # ntree is the number of trees to build
print(rf_model)

cat("\n--- Evaluating Random Forest Model Performance ---\n")
predictions_rf <- predict(rf_model, newdata = test_data_class)
results_rf <- confusionMatrix(predictions_rf, test_data_class$price_category)
print(results_rf)

# ================================================================= #
# CLASSIFICATION MODEL W/O BAGGING
# ================================================================= #

cat("\n--- Training Single Decision Tree Model ---\n")
set.seed(123)
#classic classification model "without bagging".
tree_model <- rpart(price_category ~ .,
                    data = train_data_class,
                    method = "class") # Specify 'class' for classification

cat("\n--- Visualizing the Decision Tree ---\n")
rpart.plot(tree_model)

cat("\n--- Evaluating Decision Tree Model Performance ---\n")
predictions_tree <- predict(tree_model, newdata = test_data_class, type = "class")
results_tree <- confusionMatrix(predictions_tree, test_data_class$price_category)
print(results_tree)

# ================================================================= #
#  BINARY CLASSIFICATION: PREDICTING A "HIT" GAME
# ================================================================= #

games_hit <- games_ml %>%
  filter(!is.na(playstation_score)) %>% # We can only model games that have a score
  mutate(is_hit = factor(ifelse(playstation_score >= 80, "Hit", "Not Hit")))


games_hit_train <- games_hit %>% dplyr::filter(only_year <= 2021)
games_hit_test  <- games_hit %>% dplyr::filter(only_year >= 2022)

# select predictor variables and the new 'is_hit' target variable.
train_data_hit <- games_hit_train %>%
  select(only_year, genre, playstation_score, is_hit) %>%
  separate_rows(genre, sep = "\\s*/\\s*") %>%
  na.omit()

test_data_hit <- games_hit_test %>%
  select(only_year, genre, playstation_score, is_hit) %>%
  separate_rows(genre, sep = "\\s*/\\s*") %>%
  na.omit()

# combine genre factor levels to prevent prediction errors
all_genres_hit <- unique(c(train_data_hit$genre, test_data_hit$genre))
train_data_hit$genre <- factor(train_data_hit$genre, levels = all_genres_hit)
test_data_hit$genre <- factor(test_data_hit$genre, levels = all_genres_hit)

cat("\n--- Training Binary Logistic Regression Model (Predicting 'Hit' vs 'Not Hit') ---\n")
set.seed(123)
# We use the glm() function with family = "binomial" for a standard logistic regression.
hit_model <- glm(is_hit ~ .,
                 data = train_data_hit,
                 family = "binomial")
print(summary(hit_model))

cat("\n--- Evaluating Binary Logistic Regression Model Performance ---\n")
# the model predicts the probability of a game being a "Hit"
predictions_hit_prob <- predict(hit_model, newdata = test_data_hit, type = "response")
# if the probability is greater than 0.5, we classify it as a "Hit"
predictions_hit <- factor(ifelse(predictions_hit_prob > 0.5, "Hit", "Not Hit"), levels = c("Not Hit", "Hit"))

# ensure both prediction and reference factors have the same levels to avoid errors
test_data_hit$is_hit <- factor(test_data_hit$is_hit, levels = c("Not Hit", "Hit"))

results_hit <- confusionMatrix(predictions_hit, test_data_hit$is_hit)
print(results_hit)

cat("\n--- Training Decision Tree for 'Hit' Prediction ---\n")
set.seed(123)
hit_tree_model <- rpart(is_hit ~ .,
                        data = train_data_hit,
                        method = "class")

cat("\n--- Visualizing the 'Hit' Prediction Decision Tree ---\n")
rpart.plot(hit_tree_model)

cat("\n--- Evaluating 'Hit' Prediction Decision Tree Performance ---\n")
predictions_hit_tree <- predict(hit_tree_model, newdata = test_data_hit, type = "class")

# Ensure both prediction and reference factors have the same levels
predictions_hit_tree <- factor(predictions_hit_tree, levels = c("Not Hit", "Hit"))
test_data_hit$is_hit <- factor(test_data_hit$is_hit, levels = c("Not Hit", "Hit"))

results_hit_tree <- confusionMatrix(predictions_hit_tree, test_data_hit$is_hit)
print(results_hit_tree)



