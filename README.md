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

```
# Plot 'like' click summary by month
ggplot(viz_website_2017_like_sum,
       aes(x = month, y = like_conversion_rate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1), labels = percent)
```

```
ggplot(viz_website_2018_01_sum,
       aes(x =condition, y = like_conversion_rate)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(limits = c(0, 1), labels = percent)
```
Here use `stat = "identity"` so it plots our computed values, rather than make bars of counts.
```
# Plot 'like' conversion rates by date for experiment
ggplot(viz_website_2018_02_sum,
       aes(x = visit_date,
           y = like_conversion_rate,
           color = condition,
           linetype = article_published,
           group = interaction(condition, article_published))) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = as.numeric(as.Date("2018-02-15"))) +
  scale_y_continuous(limits = c(0, 0.3), labels = percent)
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

```
# Find the p-value from the original data
disc_perm %>%
  summarize(p_value = mean(diff_orig <= stat))

# Calculate the two-sided p-value
disc_perm %>%
  summarize(p_value = mean(diff_orig <= stat) * 2)
```

`visualize` and `get_p_value` using the built in `infer` functions. Remember that the null statistics are above the original difference, so the p-value (which represents how often a null value is more extreme) is calculated by counting the number of null values which are `less` than the original difference. The small p-value indicates that the observed data are inconsistent with the null hypothesis. 

To find a two-sided p-value, you simply double the one sided p-value. That is, you want to find two times the proportion of permuted differences that are less than or equal to the observed difference.

# bootstrap

Note that because you are looking for an interval estimate, you have not made a hypothesis claim about the proportion (thus, there is no `hypothesize` step needed in the `infer` pipeline).
```
ex2_props <- all_polls %>%
  filter(poll == 1) %>%
  select(vote) %>%
  specify(response = vote, success = "yes") %>%
  generate(reps = 1000, type = "bootstrap") %>% 
  calculate(stat = "prop")
```
Many statistics we use in data analysis (including both the sample average and sample proportion) have nice properties that are used to better understand the population parameter(s) of interest.

One such property is that if the variability of the sample proportion (called the standard error, or SE) is known, then approximately 95% of p̂  values (from different samples) will be within 2SE of the true population proportion. In statistics, when sd() is applied to a variable (e.g., price of house) we call it the standard deviation. When sd() is applied to a statistic (e.g., set of sample proportions) we call it the standard error.

```
# From previous exercise: bootstrap t-confidence interval
one_poll_boot %>%
  summarize(
    lower = p_hat - 2 * sd(stat),
    upper = p_hat + 2 * sd(stat)
  )
  
# Manually calculate a 95% percentile interval
one_poll_boot %>%
  summarize(
    lower = quantile(stat, p = .025),
    upper = quantile(stat, p = .975)
  )
  
# Calculate the same interval, more conveniently
percentile_ci <- one_poll_boot %>% 
  get_confidence_interval(level = 0.95)
  
one_poll_boot %>% 
  # Visualize in-between the endpoints given by percentile_ci
  visualize(endpoints=percentile_ci, direction="between")
```

One additional element that changes the width of the confidence interval is the sample parameter value, p̂ .

Generally, when the true parameter is close to 0.5, the standard error of p̂  is larger than when the true parameter is closer to 0 or 1. When calculating a bootstrap t-confidence interval, the standard error controls the width of the CI, and here (given a true parameter of 0.8) the sample proportion is higher than in previous exercises, so the width of the confidence interval will be narrower.


# some functions

```
# Combine data from both experiments
both_ex_props <- bind_rows(ex1_props, ex2_props, .id = "experiment")
# A dataset ID column named experiment will be created.
```



# simulation

```
# Generate 10 separate random flips with probability .3
rbinom(10,1,.3)

# Generate 100 occurrences of flipping 10 coins, each with 30% probability; size = 10
rbinom(100,10,.3)

# If you flip 10 coins each with a 30% probability of coming up heads, what is the probability exactly 2 of them are heads?
dbinom(2,10,.3)

# Calculating cumulative density of a binomial. If you flip ten coins that each have a 30% probability of heads, what is the probability at least five are heads?
1 - pbinom(4, 10, .3)

# Confirm your answer with a simulation of 10,000 trials
mean(rbinom(10000, 10, .3) >= 5)
```

```
# Simulate 100,000 flips of a coin with a 40% chance of heads
A <- rbinom(100000,1,.4)

# Simulate 100,000 flips of a coin with a 20% chance of heads
B <- rbinom(100000,1,.2)

# Estimate the probability both A and B are heads
mean(A & B)

# Estimate the probability either A or B is heads
mean(A | B)
```

```
#Suppose we see 16 heads out of 20 flips, which would normally be strong evidence that the coin is biased. However, suppose we had set a prior probability of a 99% chance that the coin is fair (50% chance of heads), and only a 1% chance that the coin is biased (75% chance of heads).
# Use dbinom to find the probability of 16/20 from a fair or biased coin
probability_16_fair <- dbinom(16,20,.5)
probability_16_biased <-dbinom(16,20,.75)

# Use Bayes' theorem to find the posterior probability that the coin is fair
probability_16_fair * .99 / (probability_16_fair * .99 + probability_16_biased * .01)
# your choice of prior can have a pretty big effect on the final answer.
```

```
# Draw a random sample of 100,000 from the Binomial(1000, .2) distribution
binom_sample <- rbinom(100000, 1000, .2)

# Draw a random sample of 100,000 from the normal approximation. Remember that rnorm() takes the mean and the standard deviation, which is the square root of the variance.
normal_sample <- rnorm(100000,1000*.2, sqrt(1000*.2*.8))

# Compare the two distributions with the compare_histograms function. Remember that this takes two arguments: the first and second vectors to compare.
compare_histograms(binom_sample, normal_sample)
```

```
# Calculate the probability of <= 190 heads with pnorm
pnorm(190, 200, sqrt(160))

# Use dpois to find the exact probability that a draw is 0. Possion(2)
dpois(0, 2)
```

```
# Simulating from a Poisson and a binomial

# Draw a random sample of 100,000 from the Binomial(1000, .002) distribution
binom_sample <- rbinom(100000,1000,.002)

# Draw a random sample of 100,000 from the Poisson approximation
poisson_sample <- rpois(100000, 2)

# Compare the two distributions with the compare_histograms function
compare_histograms(binom_sample, poisson_sample)
```

One of the useful properties of the Poisson distribution is that when you add multiple Poisson distributions together, the result is also a Poisson distribution.

```
# Waiting for first coin flip
# Simulate 100 instances of flipping a 20% coin
flips <- rbinom(100,1,.2)

# Use which to find the first case of 1 ("heads")
which(flips==1)[1]

# Existing code for finding the first instance of heads
which(rbinom(100, 1, 0.2) == 1)[1]

# Replicate this 100,000 times using replicate()
replications <- replicate(100000, which(rbinom(100, 1, 0.2) == 1)[1])

# Histogram the replications with qplot
qplot(replications)

# Use the function rgeom() to simulate 100,000 draws from a geometric distributions with probability .2
geom_sample <- rgeom(100000, .2)
```

# missing values

When working with missing data, there are a couple of commands that you should be familiar with - firstly, you should be able to identify if there are any missing values, and where these are. Using the any_na() and are_na() tools, identify which values are missing.

```
# Create x, a vector, with values NA, NaN, Inf, ".", and "missing"
x <- c(NA, NaN, Inf, ".", "missing")

# Use any_na() and are_na() on to explore the missings
> any_na(x)
[1] TRUE
> are_na(x)
[1]  TRUE FALSE FALSE FALSE FALSE
```

You could use `are_na()` to and count up the missing values, but the most efficient way to count missings is to use the `n_miss()` function. This will tell you the total number of missing values in the data. You can then find the percent of missing values in the data with the `pct_miss` function. This will tell you the percentage of missing values in the data. You can also find the complement to these - how many complete values there are - using `n_complete` and `pct_complete`.

Now that you understand the behavior of missing values in R, and how to count them, let's scale up our summaries for cases (rows) and variables, using `miss_var_summary()` and `miss_case_summary()`, and also explore how they can be applied for groups in a dataframe, using the `group_by` function from `dplyr`.

```
# Return the summary of missingness in each variable, grouped by Month, in the `airquality` dataset
airquality %>% group_by(Month) %>% miss_var_summary()

# Return the summary of missingness in each case, grouped by Month, in the `airquality` dataset
airquality %>% group_by(Month) %>% miss_case_summary()
```


Another way to summarise missingness is by tabulating the number of times that there are 0, 1, 2, 3, missings in a variable, or in a case.

```
# Tabulate missingness in each variable and case of the `airquality` dataset
miss_var_table(airquality)
miss_case_table(airquality)
```

Some summaries of missingness are particularly useful for different types of data. For example, `miss_var_span()` and `miss_var_run()`.

`miss_var_span()` calculates the number of missing values in a specified variable for a repeating span. This is really useful in time series data, to look for weekly (7 day) patterns of missingness

`miss_var_run()` calculates the number of "runs" or "streaks" of missingness. This is useful to find unusual patterns of missingness, for example, you might find a repeating pattern of 5 complete and 5 missings.

```
# Calculate the summaries for each run of missingness for the variable, hourly_counts
miss_var_run(pedestrian, var = hourly_counts)

# Calculate the summaries for each span of missingness, for a span of 4000, for the variable hourly_counts
miss_var_span(pedestrian, var = hourly_counts, span_every = 4000)
```

It can be difficult to get a handle on where the missing values are in your data, and here is where visualization can really help.

The function `vis_miss()` creates an overview visualization of the missingness in the data. It also has options to cluster rows based on missingness, using `cluster = TRUE`; as well as options for sorting the columns, from most missing to least missing (`sort_miss = TRUE`).

```
# Visualize all of the missingness in the `riskfactors`  dataset
vis_miss(riskfactors)

# Visualize and cluster all of the missingness in the `riskfactors` dataset
vis_miss(riskfactors, cluster = TRUE)

# visualise and sort the columns by missingness in the `riskfactors` dataset
vis_miss(riskfactors, sort_miss = TRUE)
```

To get a clear picture of the missingness across variables and cases, use `gg_miss_var()` and `gg_miss_case()`. These are the visual counterpart to `miss_var_summary()` and `miss_case_summary()`.

These can be split up into multiple plots with one for each category by choosing a variable to facet by.

```
# Visualize the number of missings in cases using `gg_miss_case()`
gg_miss_case(riskfactors)

# Explore the number of missings in cases using `gg_miss_case()` and facet by the variable `education`
gg_miss_case(riskfactors, facet = education)

# Visualize the number of missings in variables using `gg_miss_var()`
gg_miss_var(riskfactors)

# Explore the number of missings in variables using `gg_miss_var()` and facet by the variable `education`
gg_miss_var(riskfactors, facet = education)
```

a few different ways to vizualise patterns of missingness using:

`gg_miss_upset()` to give an overall pattern of missingness.

`gg_miss_fct()` for a dataset that has a factor of interest: marriage.

and `gg_miss_span()` to explore the missingness in a time series dataset.

how to search for and count strange missing values?

```
# Explore the strange missing values "N/A"
miss_scan_count(data = pacman, search = list("N/A"))

# Explore all of the strange missing values, "N/A", "missing", "na", " "
miss_scan_count(data = pacman, search = list("N/A", "missing","na", " "))
```
replace these values with missings (e.g. `NA`) using the function `replace_with_na()`.

```
# Replace the strange missing values "N/A", "na", and "missing" with `NA` for the variables, year, and score
pacman_clean <- replace_with_na(pacman, replace = list(year = c("N/A", "na", "missing"),
                                score = c("N/A", "na", "missing")))
                                        
# Test if `pacman_clean` still has these values in it?
miss_scan_count(pacman_clean, search = list("N/A", "na", "missing"))
```

Using replace_with_na scoped variants

To reduce code repetition when replacing values with `NA`, use the "scoped variants" of `replace_with_na()`:

`replace_with_na_at()`

`replace_with_na_if()`

`replace_with_na_all()`

The syntax of replacement looks like this:

`~.x == "N/A"`

This replaces all cases that are equal to "N/A".

`~.x %in% c("N/A", "missing", "na", " ")`

Replaces all cases that have `"N/A"`, `"missing"`, `"na"`, or `" "`.

```
# Use `replace_with_na_at()` to replace with NA
replace_with_na_at(pacman,
                   .vars = c("year", "month", "day"), 
                   ~.x %in% c("N/A", "missing", "na", " "))

# Use `replace_with_na_if()` to replace with NA the character values using `is.character`
replace_with_na_if(pacman,
                   .predicate = is.character, 
                   ~.x %in% c("N/A", "missing", "na", " "))

# Use `replace_with_na_all()` to replace with NA
replace_with_na_all(pacman, ~.x %in% c("N/A", "missing", "na", " "))
```

Use the `complete()` function to make these implicit missing values explicit.

```
# Use `complete()` on the `time` and `name` variables to make implicit missing values explicit
frogger_tidy <- frogger %>% tidyr::complete(name, time)
```

One type of missing value that can be obvious to deal with is where the first entry of a group is given, but subsequent entries are marked `NA`. These missing values often result from empty values in spreadsheets to avoid entering multiple names multiple times; as well as for "human readability". This type of problem can be solved by using the `fill()` function from the `tidyr` package.

```
# Use `fill()` to fill down the name variable in the frogger dataset
frogger %>% fill(name)
```
One way to help expose missing values is to change the way we think about the data - by thinking about every single data value being missing or not missing. The `as_shadow()` function in R transforms a dataframe into a shadow matrix, a special data format where the values are either missing (`NA`), or Not Missing (`!NA`). The column names of a shadow matrix are the same as the data, but have a suffix added `_NA`. This is a useful first step in more advanced summaries of missing data.

```
# Create shadow matrix data with `as_shadow()`
as_shadow(oceanbuoys)

# Create nabular data by binding the shadow to the data with `bind_shadow()`
bind_shadow(oceanbuoys)

# Bind only the variables with missing values by using bind_shadow(only_miss = TRUE)
bind_shadow(oceanbuoys, only_miss = TRUE)
```
Let's calculate summary statistics based on the missingness of another variable.
To do this we are going to use the following steps:

First, `bind_shadow()` turns the data into nabular data.

Next, perform some summaries on the data using `group_by()` and `summarise()` to calculate the mean and standard deviation, using the `mean()` and `sd()` functions.
```
# `bind_shadow()` and `group_by()` humidity missingness (`humidity_NA`)
oceanbuoys %>%
  bind_shadow() %>%
  group_by(humidity_NA) %>% 
  summarise(wind_ew_mean = mean(wind_ew), # calculate mean of wind_ew
            wind_ew_sd = sd(wind_ew)) # calculate standard deviation of wind_ew
```

Missing values in a scatterplot in `ggplot2` are removed by default, with a warning.

We can display missing values in a scatterplot, using `geom_miss_point()` - a special `ggplot2` geom that shifts the missing values into the plot, displaying them 10% below the minimum of the variable.

```
# Use geom_miss_point() and facet_grid to explore how the missingness in wind_ew and air_temp_c is different for missingness of humidity AND by year - by using `facet_grid(humidity_NA ~ year)`
bind_shadow(oceanbuoys) %>%
  ggplot(aes(x = wind_ew,
             y = air_temp_c)) + 
  geom_miss_point() + 
  facet_grid(humidity_NA~year)
```

```
# Impute and track data with `bind_shadow`, `impute_below_all`, and `add_label_shadow`
ocean_imp_track <- bind_shadow(oceanbuoys) %>% 
  impute_below_all() %>% 
  add_label_shadow()
  
# Impute the mean value and track the imputations 
ocean_imp_mean <- bind_shadow(oceanbuoys) %>% 
  impute_mean_all() %>% 
  add_label_shadow()
```

how to explore variation across many variables and their missingness

```
# Gather the imputed data 
ocean_imp_mean_gather <- shadow_long(ocean_imp_mean,
                                     humidity,
                                     air_temp_c)
# Inspect the data
ocean_imp_mean_gather

# Explore the imputations in a histogram 
ggplot(ocean_imp_mean_gather, 
       aes(x = value, fill = value_NA)) + 
  geom_histogram() + 
  facet_wrap(~variable)
```


There are many imputation packages in R. We are going to focus on using the `simputation` package, which provides a simple, powerful interface into performing imputations.

```
# Impute humidity and air temperature using wind_ew and wind_ns, and track missing values
ocean_imp_lm_wind <- oceanbuoys %>% 
    bind_shadow() %>%
    impute_lm(air_temp_c ~ wind_ew + wind_ns) %>% 
    impute_lm(humidity ~ wind_ew + wind_ns) %>%
    add_label_shadow()
```

```
# Create the model summary for each dataset
model_summary <- bound_models %>% 
  group_by(imp_model) %>%
  nest() %>%
  mutate(mod = map(data, ~lm(sea_temp_c ~ air_temp_c + humidity + year, data = .)),
         res = map(mod, residuals),
         pred = map(mod, predict),
         tidy = map(mod, tidy))

# Explore the coefficients in the model
model_summary %>% 
	select(imp_model,tidy) %>% 
	unnest()
```

## Bayesian inference
Bayesian inference is a method for figuring out unknown or unobservable quantities given known facts. The function `prop_model` implements a Bayesian model that assumes that:

The `data` is a vector of successes and failures represented by `1`s and `0`s.

There is an unknown underlying proportion of success.

Prior to being updated with data any underlying proportion of success is equally likely.

Assume you just flipped a coin four times and the result was heads, tails, tails, heads. If you code heads as a success and tails as a failure then the following R codes runs `prop_model` with this data
```
data <- c(1, 0, 0, 1)
prop_model(data)
```
The output of prop_model is a plot showing what the model learns about the underlying proportion of success from each data point in the order you entered them. At n=0 there is no data, and all the model knows is that it's equally probable that the proportion of success is anything from 0% to 100%. At n=4 all data has been added, and the model knows a little bit more. In addition to producing a plot, `prop_model` also returns a large random sample from the posterior over the underlying proportion of success.

```
data = c(1, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0)

# Extract and explore the posterior
posterior <- prop_model(data)
head(posterior) # currently contain 10,000 samples (the default of prop_model).

# Edit the histogram
hist(posterior, breaks=30,xlim = c(0, 1),col = "palegreen4")

# to make it smoother.
# to make it cover the whole range of possible proportions from 0 to 1.
# to give it a color

# Calculate the median
median(posterior)

# Calculate the credible interval
quantile(posterior, c(0.05, 0.95))

# Calculate the probability
sum(posterior > 0.07) / length(posterior)
```
`quantile()` takes the vector of samples as its first argument and the second argument is a vector defining how much probability should be left below and above the CI. For example, the vector `c(0.05, 0.95)` would yield a 90% CI and `c(0.25, 0.75)` would yield a 50% CI.
