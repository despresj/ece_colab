library(tidyverse)
library(ggdark)
library(gganimate)


df <- read_csv("data_clean/graphics/model_convergenve.csv") %>% 
  select(-1) %>% 
  pivot_longer(cols = c(mean, min, max), 
               names_to = "stat",
               values_to = "p") %>% 
  mutate(epoch = str_c("Model Number ", epoch),
         stat = str_to_title(stat))

# plot for poster ---------------------------------------------------------

df %>%
  ggplot(aes(x = iter, y = p, linetype = stat)) +
  geom_line() +
  geom_point(size=0.2) +
  facet_wrap(~epoch, ncol = 2) +
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

animate(animated_conv, height = 9, width = 16, units = "in", res = 150)
anim_save("graphics/convergence.gif")

walk(1:5, ~beepr::beep(.x))
