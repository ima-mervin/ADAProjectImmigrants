
games <- read.csv("/Users/retajmuhsen/Desktop/masters/fall 25/MSBX 5415 (R class)/project/game_details.csv", na = c("", "--"))
library(dplyr)
library(tidyverse)
library(ipred)

#changing date to yyyy-mm-dd
games <- games |>
  mutate(release_date = as.Date(release_date, format = "%b %d, %Y"),
         year = format(release_date, "%Y"))

#changing currency to numeric
games <- games %>%
  mutate(
    highest_price = highest_price %>%
      str_replace_all("€", "") %>%
      na_if("") %>%
      as.numeric())

#removing games from 1970
games <- games %>%
  filter(year != 1970)
#splitting data
games_train <- games %>% filter(year >= 2007 & year <= 2021)
games_test  <- games %>% filter(year >= 2022 & year <= 2025)
#we removed 'Puzzle / Role playing games / Adventure' from the test data because it kept giving us an error even after 
#doing "games_test$genre <- factor(games_test$genre, levels = levels(games_train$genre))". You will see this line of code
#for question 2. Even AI could not help with this error
games_test <- games_test[games_test$genre != 'Puzzle / Role playing games / Adventure', ]

print(colSums(is.na(games)))
cat("\nOriginal number of rows:", nrow(games), "\n")

#How can we conduct a comprehensive market analysis of PlayStation game genres 
#by examining their historical popularity, prevailing pricing structures, and 
#shifting distribution over time?


#1 

#IMA HAS THE FULL CODE. USE HERS

# split the genre
split_genres <- games %>%
  # Standardize missing values
  mutate(genre = na_if(genre, "--")) %>%
  filter(!is.na(genre)) %>%
  
  # split rows based on the "/" delimiter
  # trims whitespace around each genre to avoid counting
  separate_rows(genre, sep = "\\s*/\\s*")

# accurate count of unique genres 
total_unique_genres <- split_genres %>%
  summarise(unique_genre_count = n_distinct(genre))
cat("The total unique number of individual genres is:", total_unique_genres$unique_genre_count, "\n")

## NOTE ##
## Games like "Action / Shooter" correctly contributes to the trend for both the Action market and the Shooter market. 
## This isn't double counting in a misleading way; it's accurately reflecting that the game is part of both categories.


#the graphs will be fine.

#this is a table for the trends
popularity_trends <- games %>%
  # Split mixed genres into separate rows
  separate_rows(genre, sep = "/") %>%
  mutate(genre = str_trim(genre)) %>%              # Remove spaces
  filter(genre != "--", genre != "", !is.na(genre)) %>%  # Drop invalids
  # Now group by year and genre
  group_by(year, genre) %>%
  summarise(
    avg_score = mean(playstation_score, na.rm = TRUE),
    total_ratings = sum(playstation_rating_count, na.rm = TRUE),
    .groups = "drop"
  )


#2. Relationship Between Genre and Price (Linear Regression) NOT OVER TIME, JUST PRICE AND GENRE



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
games_train$model_residuals <-residuals_log
#this gave us the error 'replacement has 2143 rows, data has 2369' so created a model with the same number of rows as residuals
log_model_fixed <- lm( log(highest_price + 1) ~ genre, data = games_train,na.action = na.exclude)

residuals_column <- residuals(log_model_fixed)

games_train$residuals <- residuals_column
