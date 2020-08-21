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
