library(tidyverse)
library(ggdark)
library(gganimate)


df <- read_csv("data_clean/graphics/model20_convergence.csv") %>% 
  select(-1) %>% 
  pivot_longer(cols = c(mean, min, max), 
               names_to = "stat",
               values_to = "p") %>% 
  mutate(epoch = str_c("Model Number ", epoch),
         stat = str_to_title(stat))

# plot for poster ---------------------------------------------------------

top <- df %>%
  group_by(epoch) %>% 
  filter(stat== "Mean") %>% 
  summarise(n = n()) %>% 
  arrange(desc(n)) %>% 
  slice_head(n = 10) %>% 
  pull(epoch)
  

df %>% 
  filter(epoch %in% top, p > 0, stat == "Min") %>% 
  ggplot(aes(x = iter, y = p, linetype = stat)) +
  geom_line() +
  geom_point(size=0.2) +
  facet_wrap(~epoch, ncol = 2, scales = "free") +
  scale_y_log10() +
  labs(title = "Convergence of GAN",
       x = "Iteration", 
       y = "Discriminator's Estimate Generated Data is Fake",
       linetype = "Summary Statistic") +
  theme_bw(base_size = 15) +
  theme(legend.position="bottom",
        strip.background =element_rect(fill="#005043"),
        strip.text = element_text(colour = 'white'))

# ggsave("poster/fig1.png", width = 16, height = 8)

# plot for slides ---------------------------------------------------------

# plot for slides ---------------------------------------------------------

animated_conv <- df %>%
  ggplot(aes(x = iter, y = p, color = stat)) +
  geom_line() +
  geom_point(size=0.2) +
  facet_wrap(~epoch, ncol = 2) +
  scale_color_viridis_d(option = 5) +
  labs(title = "Convergence of GAN",
       x = "Iteration", 
       y = "Discriminator's Estimate Generated Data is Fake",
       color = "Summary Statistic") +
  dark_theme_minimal(base_size = 15) +
  theme(legend.position="bottom") +
  transition_reveal(iter)

# animate(animated_conv, height = 9, width = 16, units = "in", res = 150)
# anim_save("graphics/convergence.gif")


library(gt)


table_data <- read_csv("data_clean/clean.csv") %>% 
  select(c(2, 4, 10, 75,76 )) %>% 
  slice(1:5) %>%
  map_df(~(.x - mean(.x)) / sd(.x))

table_data %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = " "
  ) %>% 

  gtsave("tab_1.png",  path = "graphics/"
  ) 

table_data <- table_data %>% 
  mutate(is_real = 1) 

table_data %>% 
mutate_if(is.numeric, round, 7) %>%    
  gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Label Real Data"
  ) %>% 
  data_color(is_real, color="skyblue") %>% 
  gtsave("tab_2.png",  path = "graphics/"
  )


gen_fake <- function () {
fake_data <- map(rep(5, 5), ~rnorm(.x)) %>% 
  append(list(rep(0, 5))) %>% 
  set_names(names(table_data)) %>% 
  bind_rows()
return(fake_data)
}

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Generate fake data, append to real, and train GAN"
  ) %>% 
  # data_color(is_real, color="skyblue") %>% 
  gtsave("tab_3.png",  path = "graphics/"
  )

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  select(-is_real) %>% 
  mutate(probability_is_real = c(
    runif(5, min = .95, max = 1),
    runif(5, min = 0, max = 0.05))
  ) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Estimated probability data is real"
  ) %>% 
  data_color(probability_is_real, color="#00463b") %>% 
  gtsave("tab_3_5.png",  path = "graphics/"
  )

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  select(-is_real) %>% 
  mutate(probability_is_real = c(
    runif(5, min = .95, max = 1),
    runif(5, min = 0.1, max = 0.3))
  ) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Estimated probability data is real"
  ) %>% 
  data_color(probability_is_real, color="#00463b") %>% 
  gtsave("tab_4.png",  path = "graphics/"
  )

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  select(-is_real) %>% 
  mutate(probability_is_real = c(
    runif(5, min = .95, max = 1),
    runif(5, min = .3, max = 0.5)
  )) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Estimated probability data is real"
  ) %>% 
  data_color(probability_is_real, color="#00463b") %>% 
  gtsave("tab_5.png",  path = "graphics/"
  )

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  select(-is_real) %>% 
  mutate(probability_is_real = c(
    runif(5, min = .95, max = 1),
    runif(5, min = 0.5, max = 0.9)
  )) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Estimated probability data is real"
  ) %>% 
  data_color(probability_is_real, color="#00463b") %>% 
  gtsave("tab_6.png",  path = "graphics/"
  )

fake_data <- gen_fake()

table_data %>% 
  bind_rows(fake_data) %>% 
  mutate(probability_is_real = c(
    runif(5, min = .99, max = 1),
    runif(5, min = 0.99, max = 1)
  )) %>% 
  select(-is_real) %>% 
  mutate_if(is.numeric, round, 7) %>%    gt() %>% 
  tab_header(
    title = "Minimal Example",
    subtitle = "Train GAN until generated data is indistinguishable from real data"
  ) %>% 
  data_color(probability_is_real, color="#00463b") %>% 
  gtsave("tab_7.png",  path = "graphics/"
  )
