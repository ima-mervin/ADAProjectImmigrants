# ================================================================= #
#  1. SETUP: LOAD LIBRARIES
# ================================================================= #
# install.packages(c("tidyverse", "ipred", "caret", "e1071", "stringr", "lubridate", "randomForest"))
install.packages('randomForest')

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
games <- read_csv("/Users/retajmuhsen/Desktop/masters/fall 25/MSBX 5415 (R class)/project/game_details.csv", na = c("", "--"))

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

games_train <- games %>%
  filter(only_year >= 2007 & only_year <= 2021)
games_test  <- games %>% 
  filter(only_year >= 2022 & only_year <= 2025)

#we removed the 'Puzzle / Role playing games / Adventure' genre from the test data because it kept giving us an error even after 
#doing "games_test$genre <- factor(games_test$genre, levels = levels(games_train$genre))". You will see this line of code
#for question 2. Even AI could not help with this error
games_test <- games_test[games_test$genre != 'Puzzle / Role playing games / Adventure', ]

#removing games from 1970, the data set has wrong information. The games with year '1970' werent actually released on 1970.
games <- games %>%
  filter(only_year != 1970)

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
price_genre_model <- lm(highest_price ~ genre, data = games_train)
print(summary(price_genre_model))
#no logistics 
#no classifcation tree either, cuz class tree is for trying to predict categorical stuff.

library(ggplot2)

# Histogram. looks off so we have ti log it
ggplot(games_train, aes(x = highest_price)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  labs(title = "Distribution of Game Prices", x = "Price", y = "Count")

#this looks more symmetrical 
ggplot(games_train, aes(x = log(highest_price + 1))) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Log-Transformed Game Prices",
    x = "Log(Highest Price + 1)",
    y = "Count"
  )

#factoring
games_train$genre <- factor(games_train$genre)
games_test$genre <- factor(games_test$genre, levels = levels(games_train$genre))


#model we are using for predictions 
log_model <- lm(log(highest_price + 1) ~ genre, data = games_train)
summary(log_model)


#predictions
pred_log<-predict(log_model, newdata=games_test) 
predicted_price <- exp(pred_log) - 1
print(pred_log)
print(predicted_price)

#we cant use confusion matrix because this is a regression model
#we created a scatter plot showing predicted vs actual price (using the log model)
#the red line shows how accurate the predicted vs actual prices are
games_test$predicted_price <- predicted_price
ggplot() +
  geom_point(data = games_test, aes(x = highest_price, y = predicted_price), color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Prices",
       x = "Actual Price (€)",
       y = "Predicted Price (€)")

#Residual
#In the summary for 'log_model' it shows us the Residual standard error: 0.594
residuals_log <- residuals(log_model)
residuals_log

#Creating a column for the residuals
#games_train$model_residuals <-residuals_log
#^^^^this gave us the error 'replacement has 2143 rows, data has 2369' so created a model with the same number of rows as residuals
log_model_fixed <- lm( log(highest_price + 1) ~ genre, data = games_train,na.action = na.exclude)

residuals_column <- residuals(log_model_fixed)

games_train$residuals <- residuals_column

#BOOTSTRAPING THIS MODEL
#To see if the bagging model made correct predictions, we catergorized a game as either 'high', 'low','medium'
# add the 'price_category' target variable to the main dataframe.
price_breaks <- quantile(games$highest_price, probs = c(0, 0.33, 0.66, 1.0), na.rm = TRUE)

games_train <- games_train %>%
  mutate(price_category = cut(
    highest_price,
    breaks = price_breaks,
    labels = c("Low", "Medium", "High"),
    include.lowest = TRUE
  ))

games_test <- games_test %>%
  mutate(price_category = cut(
    highest_price,
    breaks = price_breaks,   # use same quantile cutoffs
    labels = c("Low", "Medium", "High"),
    include.lowest = TRUE
  ))

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

#Now we can bag the categorical prices (high, low, medium)
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



# --- Question 3: How has the distribution of genres on PlayStation changed over time? ---
cat("\n--- 3. Distribution of Genres Over Time ---\n")
genre_trends <- games %>%
  filter(!is.na(only_year)) %>%
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
                aes(x = only_year, y = n, group = genre, color = genre)) +
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
games_train <- games_ml %>% dplyr::filter(only_year <= 2021)
games_test  <- games_ml %>% dplyr::filter(only_year >= 2022)

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
#  BAGGING. I PUT THIS IN QUESTION 2.
# ================================================================= #
#cat("\n--- Training Bagging Classification Model ---\n")
#set.seed(123) # for reproducibility
#bagging_model <- bagging(price_category ~ ., # `.` means 'use all other columns as predictors'
                         #data = train_data_class,
                         #nbagg = 50)
#print(bagging_model)

#cat("\n--- Evaluating Bagging Model Performance ---\n")
#predictions_bagging <- predict(bagging_model, newdata = test_data_class)
#results_bagging <- confusionMatrix(predictions_bagging, test_data_class$price_category)
#print(results_bagging)

# ================================================================= #
# RANDOM FOREST. This is a different version of bagging. not going to use it for now.
# ================================================================= #

#cat("\n--- Training Random Forest Model ---\n")
#set.seed(123)
#rf_model <- randomForest(price_category ~ .,
                         #data = train_data_class,
                         #ntree = 100) # ntree is the number of trees to build
#print(rf_model)

#cat("\n--- Evaluating Random Forest Model Performance ---\n")
#predictions_rf <- predict(rf_model, newdata = test_data_class)
#results_rf <- confusionMatrix(predictions_rf, test_data_class$price_category)
#print(results_rf)

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




