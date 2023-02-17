data <- read_csv("canny_edge_detector_timings.csv")
data$time <- data$time/1000

data %>%
  ggplot(aes(optimization, time, fill = openMP)) +
  geom_boxplot() +
  labs(title = "Execution Times vs Optimization Levels With/Without OpenMP", x = "optimization level", y = "time (ms)")
ggsave("boxplot.png")

data %>%
  filter(pgm == "cat") %>%
  ggplot(aes(time, fill = openMP)) +
  geom_histogram(position = "dodge") +
  labs(title = "Frequency of Execution Time With/Without OpenMP on cat.pgm", x = "time (ms)", y = "Frequency")
ggsave("histogram_cat.png")

data %>%
  filter(pgm == "galaxy") %>%
  ggplot(aes(time, fill = openMP)) +
  geom_histogram(position = "dodge") +
  labs(title = "Frequency of Execution Time With/Without OpenMP on galaxy.ascii.pgm", x = "time (ms)", y = "Frequency")
ggsave("histogram_galaxy.png")