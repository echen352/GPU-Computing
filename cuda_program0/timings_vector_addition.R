data <- read_csv("timings.csv")
data$newSize <- as.factor(data$size)
p <- data %>%
  ggplot(aes(newSize, time, color = cuda)) +
  geom_point() +
  ylab("time (sec) log scale") +
  xlab("Vector Size") +
  labs(title = "Vector Addition Cuda vs Non-Cuda Performance")
p + scale_y_log10()
