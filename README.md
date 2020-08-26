# DataCamp-R

Parallel slopes models are so-named because we can visualize these models in the data space as not one line, but two parallel lines. To do this, we'll draw two things:

a scatterplot showing the data, with color separating the points into groups

a line for each value of the categorical variable

```
# Augment the model. The augment() function from the broom package provides an easy way to add the fitted values to our data frame
augmented_mod <- augment(mod)
glimpse(augmented_mod)

# scatterplot, with color
data_space <- ggplot(data=augmented_mod, aes(x = wheels, y = totalPr, color = cond)) + 
  geom_point()
  
# single call to geom_line()
data_space + 
  geom_line(aes(y = .fitted))
  
# Note that this approach has the added benefit of automatically coloring the lines appropriately to match the data.  
```

Use geom_smooth() to add the logistic regression line

```
# scatterplot with jitter
data_space <- ggplot(data = MedGPA, aes(y = Acceptance, x = GPA)) + 
  geom_jitter(width = 0, height = 0.05, alpha = 0.5)

# add logistic curve
data_space +
  geom_smooth(method = "glm", se = FALSE, method.args = list(family = "binomial"))
```

We need to tell the glm() function which member of the GLM family we want to use. To do this, we will pass the family argument to glm() as a list using the method.args argument to geom_smooth(). This mechanism is common in R, and allows one function to pass a list of arguments to another function.


```
# augmented model
MedGPA_plus <- mod %>%
  augment(type.predict = "response")
```

One quick technique for jump-starting exploratory data analysis (EDA)  is to examine all of the pairwise scatterplots in your data. This can be achieved using the pairs() function. pairs(df)

Use the colnames() function to list the variables included in data frame. colnames(df)

# ggplot options
```
# Density plot of SleepHrsNight colored by SleepTrouble
ggplot(NHANES, aes(x = SleepHrsNight, color = SleepTrouble)) + 
  # Adjust by 2; Since SleepHrsNight contains discrete values, the density should be smoothed a bit using adjust = 2.
  geom_density(adjust = 2) + 
  # Facet by HealthGen
  facet_wrap(~ HealthGen)
```
# Calculating statistic of interest

```
fruits <- c("apple", "banana", "cherry")
fruits %in% c("banana", "cherry")
mean(fruits %in% c("banana", "cherry"))

diff_orig <- homes %>%   
  # Group by gender
  group_by(Gender) %>%
  # Summarize proportion of homeowners
  summarize(prop_own = mean(HomeOwn == "Own")) %>%
  # Summarize difference in proportion of homeowners
  summarize(obs_diff_prop = diff(prop_own)) # male - female
```
# Randomized data under null model of independence
The infer package will allow you to model a particular null hypothesis and then randomize the data to calculate permuted statistics. In this exercise, after specifying your null hypothesis you will permute the home ownership variable 10 times. By doing so, you will ensure that there is no relationship between home ownership and gender, so any difference in home ownership proportion for female versus male will be due only to natural variability.

This exercise will demonstrate the four steps from the infer package:

`specify` will specify the response and explanatory variables.

`hypothesize` will declare the null hypothesis.

`generate` will generate resamples, permutations, or simulations.

`calculate` will calculate summary statistics.

```
# Perform 1000 permutations
homeown_perm <- homes %>%
# Specify HomeOwn vs. Gender, with `"Own" as success
  specify(HomeOwn ~ Gender, success = "Own") %>%
# Use a null hypothesis of independence
  hypothesize(null = "independence") %>% 
# Generate 1000 repetitions (by permutation)
  generate(reps = 1000, type = "permute") %>% 
# Calculate the difference in proportions (male then female)
  calculate(stat = "diff in props", order = c("male", "female"))
  
# Dotplot of 1000 permuted differences in proportions
ggplot(homeown_perm, aes(x = stat)) + 
  geom_dotplot(binwidth=0.001)  
  
# Density plot of 1000 permuted differences in proportions
ggplot(homeown_perm, aes(x = stat)) + 
  geom_density()
```

It is important to know whether any of the randomly permuted differences were as extreme as the observed difference. Using the specify-hypothesis-generate-calculate workflow in infer, you can calculate the same statistic, but instead of getting a single number, you get a whole distribution.
```
# Plot permuted differences, diff_perm
ggplot(homeown_perm, aes(x = diff_perm)) + 
  # Add a density layer
  geom_density() +
  # Add a vline layer with intercept diff_orig
  geom_vline(aes(xintercept = diff_orig), color = "red")

# Compare permuted differences to observed difference
homeown_perm %>%
  summarize(n_perm_le_obs = sum(diff_perm <= diff_orig))
```  
```  
# Create a contingency table summarizing the data
disc %>%
  # Count the rows by sex, promote
  count(sex, promote)

# Find proportion of each sex who were promoted
disc %>%
  # Group by sex
  group_by(sex) %>%
  # Calculate proportion promoted summary stat
  summarize(promoted_prop = mean(promote == "promoted"))
```
Note that with binary variables, the proportion of either value can be found using the mean() function (e.g. mean(variable == "value")).

To quantify the extreme permuted (null) differences, we use the quantile() function.

```
disc_perm %>% 
  summarize(
    # Find the 0.9 quantile of diff_perm's stat
    q.90 = quantile(stat, p = 0.9)
  )
```

```
# Visualize and calculate the p-value for the original dataset. visualize() and get_p_value() using the built in infer functions. 
disc_perm %>%
  visualize(obs_stat = diff_orig, direction = "greater")
  
disc_perm %>%
  get_p_value(diff_orig, "greater")
```
